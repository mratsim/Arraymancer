# Copyright 2017 Mamy André-Ratsimbazafy
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import  ./backend/openmp,
        ./private/p_checks,
        ./data_structure, ./init_cpu, ./accessors,
        future

# Note: Due to C++ restrictions + Nim codegen, mutable iterators are not possible with the C++ backend.
# Cuda and C++ need a specific code path (with no performance implication since what was done in the iterator
# is no done in the higher order function/template context)

# ####################################################################
# Mapping over tensors

template apply_inline*(t: var Tensor, op: untyped): untyped =
  omp_parallel_blocks(block_offset, block_size, t.size):
    for x {.inject.} in t.mitems(block_offset, block_size):
      x = op

template apply2_inline*[T,U](dest: var Tensor[T], src: Tensor[U], op: untyped): untyped =
  omp_parallel_blocks(block_offset, block_size, dest.size):
    for x {.inject.}, y {.inject.} in mzip(dest, src, block_offset, block_size):
      x = op

template apply3_inline*[T,U,V](dest: var Tensor[T], src1: Tensor[U], src2: Tensor[V], op: untyped): untyped =
  omp_parallel_blocks(block_offset, block_size, dest.size):
    for x {.inject.}, y {.inject.}, z {.inject.} in mzip(dest, src1, src2, block_offset, block_size):
      x = op

template map_inline*[T](t: Tensor[T], op:untyped): untyped =
  var dest = newTensorUninit[T](t.shape)
  var data = dest.dataArray
  omp_parallel_blocks(block_offset, block_size, dest.size):
    for i, x {.inject.} in enumerate(t, block_offset, block_size):
      data[i] = op
  dest.unsafeView()

template map2_inline*[T, U](t1: Tensor[T], t2: Tensor[U], op:untyped): untyped =
  when compileOption("boundChecks"):
    check_elementwise(t1,t2)

  var dest = newTensorUninit[T](t1.shape)
  var data = dest.dataArray
  omp_parallel_blocks(block_offset, block_size, t1.size):
    for i, x {.inject.}, y {.inject.} in enumerateZip(t1, t2, block_offset, block_size):
      data[i] = op
  dest.unsafeView()

template map3_inline*[T, U, V](t1: Tensor[T], t2: Tensor[U], t3: Tensor[V], op:untyped): untyped =
  when compileOption("boundChecks"):
    check_elementwise(t1,t2)
    check_elementwise(t1,t3)

  var dest = newTensorUninit[T](t1.shape)
  var data = dest.dataArray
  omp_parallel_blocks(block_offset, block_size, t1.size):
    for i, x {.inject.}, y {.inject.}, z {.inject.} in enumerateZip(t1, t2, t3, block_offset, block_size):
      data[i] = op
  dest.unsafeView()

template reduce_inline*[T](t: Tensor[T], op: untyped): untyped =
  var reduced : T
  omp_parallel_reduce_blocks(reduced, block_offset, block_size, t.size, 1, op) do:
    x = t.atContiguousIndex(block_offset)
  do:
    for y {.inject.} in t.items(block_offset, block_size):
      op
  reduced

template fold_inline*[T](t: Tensor[T], op_initial, op_middle, op_final: untyped): untyped =
  var reduced : T
  omp_parallel_reduce_blocks(reduced, block_offset, block_size, t.size, 1, op_final) do:
    let y {.inject.} = t.atContiguousIndex(block_offset)
    op_initial
  do:
    for y {.inject.} in t.items(block_offset, block_size):
      op_middle
  reduced

template reduce_axis_inline*[T](t: Tensor[T], reduction_axis: int, op: untyped): untyped =
  var reduced : type(t)
  let weight = t.size div t.shape[reduction_axis]
  omp_parallel_reduce_blocks(reduced, block_offset, block_size, t.shape[reduction_axis], weight, op) do:
    x = t.atAxisIndex(reduction_axis, block_offset).unsafeView()
  do:
    for y {.inject.} in t.axis(reduction_axis, block_offset, block_size):
      op
  reduced.unsafeView()

template fold_axis_inline*[T](t: Tensor[T], fold_axis: int, op_initial, op_middle, op_final: untyped): untyped =
  var reduced : type(t)
  let weight = t.size div t.shape[fold_axis]
  omp_parallel_reduce_blocks(reduced, block_offset, block_size, t.shape[fold_axis], weight, op_final) do:
    let y {.inject.} = t.atAxisIndex(fold_axis, block_offset).unsafeView()
    op_initial
  do:
    for y {.inject.} in t.axis(fold_axis, block_offset, block_size):
      op_middle
  reduced.unsafeView()

proc map*[T; U: not (ref|string|seq)](t: Tensor[T], f: T -> U): Tensor[U] {.noInit.} =
  ## Apply a unary function in an element-wise manner on Tensor[T], returning a new Tensor.
  ## Usage with Nim's ``future`` module:
  ##  .. code:: nim
  ##     a.map(x => x+1) # Map the anonymous function x => x+1
  ## Usage with named functions:
  ##  .. code:: nim
  ##     proc plusone[T](x: T): T =
  ##       x + 1
  ##     a.map(plusone) # Map the function plusone
  ## Note:
  ##   for basic operation, you can use implicit broadcasting instead
  ##   with operators prefixed by a dot :
  ##  .. code:: nim
  ##     a .+ 1
  ## ``map`` is especially useful to do multiple element-wise operations on a tensor in a single loop over the data.
  ##
  ## For OpenMP compatibility, this ``map`` doesn't allow ref types as result like seq or string

  result = newTensorUninit[U](t.shape)
  result.apply2_inline(t, f(y))

proc map*[T; U: ref|string|seq](t: Tensor[T], f: T -> U): Tensor[U] {.noInit,noSideEffect.}=
  ## Apply a unary function in an element-wise manner on Tensor[T], returning a new Tensor.
  ##
  ##
  ## This is a fallback for ref types as
  ## OpenMP will not work with if the results allocate memory managed by GC.

  result = newTensorUninit[U](t.shape)
  for i, val in enumerate(t):
    result.data[i] = f(val)

proc apply*[T](t: var Tensor[T], f: T -> T) =
  ## Apply a unary function in an element-wise manner on Tensor[T], in-place.
  ##
  ## Input:
  ##   - a var Tensor
  ##   - A function or anonymous function that ``returns a value``
  ## Result:
  ##   - Nothing, the ``var`` Tensor is modified in-place
  ##
  ## Usage with Nim's ``future`` module:
  ##  .. code:: nim
  ##     var a = newTensor([5,5], int) # a must be ``var``
  ##     a.apply(x => x+1) # Map the anonymous function x => x+1
  ## Usage with named functions:
  ##  .. code:: nim
  ##     proc plusone[T](x: T): T =
  ##       x + 1
  ##     a.apply(plusone) # Apply the function plusone in-place

  t.apply_inline(f(x))

proc apply*[T](t: var Tensor[T], f: proc(x:var T)) =
  ## Apply a unary function in an element-wise manner on Tensor[T], in-place.
  ##
  ## Input:
  ##   - a var Tensor
  ##   - An in-place function that ``returns no value``
  ## Result:
  ##   - Nothing, the ``var`` Tensor is modified in-place
  ##
  ## Usage with Nim's ``future`` module:
  ##   - Future module has a functional programming paradigm, anonymous function cannot mutate
  ##     the arguments
  ## Usage with named functions:
  ##  .. code:: nim
  ##     proc pluseqone[T](x: var T) =
  ##       x += 1
  ##     a.apply(pluseqone) # Apply the in-place function pluseqone
  ## ``apply`` is especially useful to do multiple element-wise operations on a tensor in a single loop over the data.

  omp_parallel_blocks(block_offset, block_size, t.size):
    for x in t.mitems(block_offset, block_size):
      f(x)

proc map2*[T, U; V: not (ref|string|seq)](t1: Tensor[T],
                                          f: (T,U) -> V,
                                          t2: Tensor[U]): Tensor[V] {.noInit.} =
  ## Apply a binary function in an element-wise manner on two Tensor[T], returning a new Tensor.
  ##
  ## The function is applied on the elements with the same coordinates.
  ##
  ## Input:
  ##   - A tensor
  ##   - A function
  ##   - A tensor
  ## Result:
  ##   - A new tensor
  ## Usage with named functions:
  ##  .. code:: nim
  ##     proc `**`[T](x, y: T): T = # We create a new power `**` function that works on 2 scalars
  ##       pow(x, y)
  ##     a.map2(`**`, b)
  ##     # Or
  ##     map2(a, `**`, b)
  ## ``map2`` is especially useful to do multiple element-wise operations on a two tensors in a single loop over the data.
  ## for example ```alpha * sin(A) + B```
  ##
  ## For OpenMP compatibility, this ``map2`` doesn't allow ref types as result like seq or string
  when compileOption("boundChecks"):
    check_elementwise(t1,t2)

  result = newTensorUninit[V](t1.shape)
  result.apply3_inline(t1, t2, f(y,z))

proc map2*[T, U; V: ref|string|seq](t1: Tensor[T],
                                    f: (T,U) -> V,
                                    t2: Tensor[U]): Tensor[V] {.noInit,noSideEffect.}=
  ## Apply a binary function in an element-wise manner on two Tensor[T], returning a new Tensor.
  ##
  ##
  ## This is a fallback for ref types as
  ## OpenMP will not work with if the results allocate memory managed by GC.
  when compileOption("boundChecks"):
    check_elementwise(t1,t2)

  result = newTensorUninit[V](t1.shape)
  for r, a, b in mzip(result, t1, t2):
    r = f(t1, t2)

proc apply2*[T, U](a: var Tensor[T],
                   f: proc(x:var T, y:T), # We can't use the nice future syntax here
                   b: Tensor[U]) =
  ## Apply a binary in-place function in an element-wise manner on two Tensor[T], returning a new Tensor.
  ##
  ## The function is applied on the elements with the same coordinates.
  ##
  ## Input:
  ##   - A var tensor
  ##   - A function
  ##   - A tensor
  ## Result:
  ##   - Nothing, the ``var``Tensor is modified in-place
  ## Usage with named functions:
  ##  .. code:: nim
  ##     proc `**=`[T](x, y: T) = # We create a new in-place power `**=` function that works on 2 scalars
  ##       x = pow(x, y)
  ##     a.apply2(`**=`, b)
  ##     # Or
  ##     apply2(a, `**=`, b)
  ## ``apply2`` is especially useful to do multiple element-wise operations on a two tensors in a single loop over the data.
  ## for example ```A += alpha * sin(A) + B```
  when compileOption("boundChecks"):
    check_elementwise(a,b)

  omp_parallel_blocks(block_offset, block_size, a.size):
    for x, y in mzip(a, b, block_offset, block_size):
      f(x, y)

# ####################################################################
# Folds and reductions over a single Tensor

# Note: You can't pass builtins like `+` or `+=` due to Nim limitations
# https://github.com/nim-lang/Nim/issues/2172

proc fold*[U, T](t: Tensor[U],
                start_val: T,
                f:(T, U) -> T,
                ): T =
  ## Chain result = f(result, element) over all elements of the Tensor
  ## Input:
  ##     - A tensor to aggregate on
  ##     - The starting value
  ##     - The aggregation function. It is applied this way: new_aggregate = f(old_aggregate, current_value)
  ## Result:
  ##     - An aggregate of the function called on the starting value and all elements of the tensor
  ## Usage:
  ##  .. code:: nim
  ##     a.fold(100,max) ## This compare 100 with the first tensor value and returns 100
  ##                     ## In the end, we will get the highest value in the Tensor or 100
  ##                     ## whichever is bigger.

  result = start_val
  for val in t:
    result = f(result, val)

proc fold*[U, T](t: Tensor[U],
                start_val: Tensor[T],
                f: (Tensor[T], Tensor[U])-> Tensor[T],
                axis: int
                ): Tensor[T] =
  ## Chain result = f(result, element) over all elements of the Tensor
  ## Input:
  ##     - A tensor to aggregate on
  ##     - The starting value
  ##     - The aggregation function. It is applied this way: new_aggregate = f(old_aggregate, current_value)
  ##     - The axis to aggregate on
  ## Result:
  ##     - An Tensor with the aggregate of the function called on the starting value and all slices along the selected axis

  result = start_val
  for val in t.axis(axis):
    result = f(result, val)

proc reduce*[T](t: Tensor[T],
                f: (T, T) -> T
                ): T =
  ## Chain result = f(result, element) over all elements of the Tensor.
  ##
  ## The starting value is the first element of the Tensor.
  ## Input:
  ##     - A tensor to aggregate on
  ##     - The aggregation function. It is applied this way: new_aggregate = f(old_aggregate, current_value)
  ## Result:
  ##     - An aggregate of the function called all elements of the tensor
  ## Usage:
  ##  .. code:: nim
  ##     a.reduce(max) ## This returns the maximum value in the Tensor.

  t.reduce_inline():
    shallowCopy(x, f(x,y))

proc reduce*[T](t: Tensor[T],
                f: (Tensor[T], Tensor[T])-> Tensor[T],
                axis: int
                ): Tensor[T] {.noInit.} =
  ## Chain result = f(result, element) over all elements of the Tensor.
  ##
  ## The starting value is the first element of the Tensor.
  ## Input:
  ##     - A tensor to aggregate on
  ##     - The aggregation function. It is applied this way: new_aggregate = f(old_aggregate, current_value)
  ##     - An axis to aggregate on
  ## Result:
  ##     - A tensor aggregate of the function called all elements of the tensor

  t.reduce_axis_inline(axis):
    shallowCopy(x, f(x,y))
