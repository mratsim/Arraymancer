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


#####################################################################
# Mapping over tensors

proc map*[T, U](t: Tensor[T], f: T -> U): Tensor[U] {.noSideEffect.}=
  ## Map a unary function T -> U on Tensor[T]

  # We use this opportunity to reshape the data internally
  # Iteration should be almost as fast for contiguous non-sliced Tensors
  # But may avoid a lot of unnecessary computations on slices
  tensorCpu(t.shape, result)

  result.data = newSeq[U](result.size)
  var i = 0 # TODO: use pairs/enumerate instead - pending https://forum.nim-lang.org/t/2972
  for val in t:
    result.data[i] = f(val)
    inc i

proc apply*[T](t: var Tensor[T], f: T -> T) {.noSideEffect.}=
  ## Map an unary function T->T in place to all elements of the Tensor.
  for val in t.mitems:
    val = f(val)

proc apply*[T](t: var Tensor[T], f: proc(x:var T)) {.noSideEffect.}=
  ## Map an unary function T -> nil in place to all elements of the Tensor.
  for val in t.mitems:
    f(val)

proc map2*[T, U, V](t1: Tensor[T], f: (T,U) -> V, t2: Tensor[U]): Tensor[V] {.noSideEffect.}=
  ## Map a binary function (T,U) -> V on 2 tensors
  ## It applies the function to each matching elements
  ## Tensors must have the same shape
  ##
  ## Note the new argument order of map2 to accomodate for
  ## t1.map2(`op`, t2) where op is an infix operator.

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
  ## Apply element-wise a binary function (T,U)->T like a += b
  ## Result is stored inplace in the first tensor
  ## Note: builtin functions like `+=` cannot be used as is as function argument
  when compileOption("boundChecks"):
    check_elementwise(a,b)

  ## TODO: yield mutable values for a: https://forum.nim-lang.org/t/2972
  for a_idx, b_val in zip(a.real_indices, b.values):
    f(a.data[a_idx], b_val)

#####################################################################
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

  result = start_val
  for val in t:
    result = f(result, val)

proc fold*[U, T](t: Tensor[U],
                start_val: Tensor[T],
                f: (Tensor[T], Tensor[U])-> Tensor[T],
                axis: int
                ): Tensor[T] {.noSideEffect.}=
  ## Chain result = f(result, element) over an axis of the Tensor
  ## Input:
  ##     - A tensor to aggregate on
  ##     - The starting value
  ##     - The aggregation function. It is applied this way: new_aggregate = f(old_aggregate, current_value)
  ##     - The axis

  result = start_val
  for val in t.axis(axis):
    result = f(result, val)

proc fold2*[T](t1: Tensor[T],
               start_val: T,
               f: (T, T, T)-> T,
               t2: Tensor[T]
               ): T {.noSideEffect.}=
  ## Chain result = f(result, element1, element2) over all elements of two Tensors
  ## Input:
  ##     - Two tensor to aggregate on
  ##     - The starting value
  ##     - The aggregation function. It is applied this way: new_aggregate = f(old_aggregate, element1, element2)
  result = start_val
  for ai, bi in zip(t1.values, t2.values):
    result = f(result, ai, bi)

proc reduce*[T](t: Tensor[T],
                f: (T, T) -> T
                ): T {.noSideEffect.}=
  ## Chain result = f(result, element) over all elements of the Tensor
  ## The starting value is the first element of the Tensor.
  ## Input:
  ##     - A tensor to aggregate on
  ##     - The aggregation function. It is applied this way: new_aggregate = f(old_aggregate, current_value)

  let it = t.values
  result = it()
  for val in it():
    result = f(result, val)

proc reduce*[T](t: Tensor[T],
                f: (Tensor[T], Tensor[T])-> Tensor[T],
                axis: int
                ): Tensor[T] {.noSideEffect.}=
  ## Chain result = f(result, element) over all elements of the Tensor
  ## The starting value is the first element on the axis of the Tensor.
  ## Input:
  ##     - A tensor to aggregate on
  ##     - The aggregation function. It is applied this way: new_aggregate = f(old_aggregate, current_value)
  ##     - The axis to aggregate on

  let it = t.axis(axis)
  result = it()
  for val in it():
    result = f(result, val)