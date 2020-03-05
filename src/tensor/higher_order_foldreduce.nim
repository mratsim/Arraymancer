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
        ./data_structure, ./init_copy_cpu, ./accessors,
        sugar

template reduce_inline*[T](t: Tensor[T], op: untyped): untyped =
  let z = t # ensure that if t is the result of a function it is not called multiple times
  var reduced: T
  omp_parallel_reduce_blocks(reduced, block_offset, block_size, z.size, 1, op) do:
    x = z.atContiguousIndex(block_offset)
  do:
    for y {.inject.} in z.items(block_offset, block_size):
      op
  reduced

template fold_inline*[T](t: Tensor[T], op_initial, op_middle, op_final: untyped): untyped =
  let z = t # ensure that if t is the result of a function it is not called multiple times
  var reduced: T
  omp_parallel_reduce_blocks(reduced, block_offset, block_size, z.size, 1, op_final) do:
    let y {.inject.} = z.atContiguousIndex(block_offset)
    op_initial
  do:
    for y {.inject.} in z.items(block_offset, block_size):
      op_middle
  reduced

template reduce_axis_inline*[T](t: Tensor[T], reduction_axis: int, op: untyped): untyped =
  let z = t # ensure that if t is the result of a function it is not called multiple times
  var reduced: type(z)
  let weight = z.size div z.shape[reduction_axis]
  omp_parallel_reduce_blocks(reduced, block_offset, block_size, z.shape[reduction_axis], weight, op) do:
    x = z.atAxisIndex(reduction_axis, block_offset).clone()
  do:
    for y {.inject.} in z.axis(reduction_axis, block_offset, block_size):
      op
  reduced

template fold_axis_inline*[T](t: Tensor[T], accumType: typedesc, fold_axis: int, op_initial, op_middle, op_final: untyped): untyped =
  let z = t # ensure that if t is the result of a function it is not called multiple times
  var reduced: accumType
  let weight = z.size div z.shape[fold_axis]
  omp_parallel_reduce_blocks(reduced, block_offset, block_size, z.shape[fold_axis], weight, op_final) do:
    let y {.inject.} = z.atAxisIndex(fold_axis, block_offset).clone()
    op_initial
  do:
    for y {.inject.} in z.axis(fold_axis, block_offset, block_size):
      op_middle
  reduced

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
                f: (Tensor[T], Tensor[U]) -> Tensor[T],
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
                f: (Tensor[T], Tensor[T]) -> Tensor[T],
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
