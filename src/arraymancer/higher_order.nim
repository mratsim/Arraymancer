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

proc check_elementwise(a, b:AnyTensor)  {.noSideEffect.}=
  ## Check if element-wise operations can be applied to 2 Tensors
  if a.shape != b.shape:
    raise newException(ValueError, "Both Tensors should have the same shape")


# ####################################################################
# Mapping over tensors

proc map*[T, U](t: Tensor[T], f: T -> U): Tensor[U] {.noSideEffect.}=
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

  # We use this opportunity to reshape the data internally to contiguous.
  # Iteration should be almost as fast for contiguous non-sliced Tensors
  # And should benefit future computations on previously non-contiguous data
  tensorCpu(t.shape, result)

  result.data = newSeq[U](result.size)
  var i = 0 # TODO: use pairs/enumerate instead - pending https://forum.nim-lang.org/t/2972
  for val in t:
    result.data[i] = f(val)
    inc i

proc apply*[T](t: var Tensor[T], f: T -> T) {.noSideEffect.}=
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
  for val in t.mitems:
    val = f(val)

proc apply*[T](t: var Tensor[T], f: proc(x:var T)) {.noSideEffect.}=
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
  for val in t.mitems:
    f(val)

proc map2*[T, U, V](t1: Tensor[T], f: (T,U) -> V, t2: Tensor[U]): Tensor[V] {.noSideEffect.}=
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
  when compileOption("boundChecks"):
    check_elementwise(t1,t2)

  tensorCpu(t1.shape, result)

  result.data = newSeq[U](result.size)

  # TODO use mitems instead of result.data[i] cf profiling
  # TODO: inline iterators - pending https://github.com/nim-lang/Nim/issues/4516
  for i, ai, bi in enumerate_zip(t1.values, t2.values):
    result.data[i] = f(ai, bi)

proc apply2*[T, U](a: var Tensor[T],
                   f: proc(x:var T, y:T), # We can't use the nice future syntax here
                   b: Tensor[U]) {.noSideEffect.}=
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

  ## TODO: yield mutable values for a: https://forum.nim-lang.org/t/2972
  for a_idx, b_val in zip(a.real_indices, b.values):
    f(a.data[a_idx], b_val)

# ####################################################################
# Folds and reductions over a single Tensor

# Note: You can't pass builtins like `+` or `+=` due to Nim limitations
# https://github.com/nim-lang/Nim/issues/2172

proc fold*[U, T](t: Tensor[U],
                start_val: T,
                f:(T, U) -> T,
                ): T {.noSideEffect.}=
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
                ): Tensor[T] {.noSideEffect.}=
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
                ): T {.noSideEffect.}=
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

  let it = t.values
  result = it()
  for val in it():
    result = f(result, val)

proc reduce*[T](t: Tensor[T],
                f: (Tensor[T], Tensor[T])-> Tensor[T],
                axis: int
                ): Tensor[T] {.noSideEffect.}=
  ## Chain result = f(result, element) over all elements of the Tensor.
  ##
  ## The starting value is the first element of the Tensor.
  ## Input:
  ##     - A tensor to aggregate on
  ##     - The aggregation function. It is applied this way: new_aggregate = f(old_aggregate, current_value)
  ##     - An axis to aggregate on
  ## Result:
  ##     - A tensor aggregate of the function called all elements of the tensor

  let it = t.axis(axis)
  result = it()
  for val in it():
    result = f(result, val)