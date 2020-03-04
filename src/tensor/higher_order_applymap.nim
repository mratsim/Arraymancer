# Copyright 2017 the Arraymancer contributors
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
        sugar

# ####################################################################
# Mapping over tensors

template apply_inline*(t: var Tensor, op: untyped): untyped =
  # TODO, if t is a result of a function
  # how to ensure that it is not called twice
  omp_parallel_blocks(block_offset, block_size, t.size):
    for x {.inject.} in t.mitems(block_offset, block_size):
      x = op

template apply2_inline*[T,U](dest: var Tensor[T], src: Tensor[U], op: untyped): untyped =
  # TODO, if dest is a result of a function
  # how to ensure that it is not called twice
  omp_parallel_blocks(block_offset, block_size, dest.size):
    for x {.inject.}, y {.inject.} in mzip(dest, src, block_offset, block_size):
      x = op

template apply3_inline*[T,U,V](dest: var Tensor[T], src1: Tensor[U], src2: Tensor[V], op: untyped): untyped =
  # TODO, if dest is a result of a function
  # how to ensure that it is not called twice
  omp_parallel_blocks(block_offset, block_size, dest.size):
    for x {.inject.}, y {.inject.}, z {.inject.} in mzip(dest, src1, src2, block_offset, block_size):
      x = op

template map_inline*[T](t: Tensor[T], op:untyped): untyped =

  let z = t # ensure that if t is the result of a function it is not called multiple times

  type outType = type((
    block:
      var x{.inject.}: type(items(z));
      op
  ))

  var dest = newTensorUninit[outType](z.shape)
  let data = dest.unsafe_raw_data()

  omp_parallel_blocks(block_offset, block_size, dest.size):
    for i, x {.inject.} in enumerate(z, block_offset, block_size):
      data[i] = op
  dest

template map2_inline*[T, U](t1: Tensor[T], t2: Tensor[U], op:untyped): untyped =

  let
    z1 = t1 # ensure that if t1 is the result of a function it is not called multiple times
    z2 = t2

  when compileOption("boundChecks"):
    check_elementwise(z1,z2)

  type outType = type((
    block:
      var x{.inject.}: type(items(z1));
      var y{.inject.}: type(items(z2));
      op
  ))

  var dest = newTensorUninit[outType](z1.shape)
  let data = dest.unsafe_raw_data()

  omp_parallel_blocks(block_offset, block_size, z1.size):
    for i, x {.inject.}, y {.inject.} in enumerateZip(z1, z2, block_offset, block_size):
      data[i] = op
  dest

template map3_inline*[T, U, V](t1: Tensor[T], t2: Tensor[U], t3: Tensor[V], op:untyped): untyped =
  let
    z1 = t1 # ensure that if t1 is the result of a function it is not called multiple times
    z2 = t2
    z3 = t3

  when compileOption("boundChecks"):
    check_elementwise(z1,z2)
    check_elementwise(z1,z3)

  type outType = type((
    block:
      var x{.inject.}: type(items(z1));
      var y{.inject.}: type(items(z2));
      var z{.inject.}: type(items(z3));
      op
  ))

  var dest = newTensorUninit[outType](z1.shape)
  let data = dest.unsafe_raw_data()

  omp_parallel_blocks(block_offset, block_size, z1.size):
    for i, x {.inject.}, y {.inject.}, z {.inject.} in enumerateZip(z1, z2, z3, block_offset, block_size):
      data[i] = op
  dest

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
  ##     a +. 1
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
