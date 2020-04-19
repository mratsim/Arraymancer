# Copyright 2017-2020 Mamy-André Ratsimbazafy
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

import  ./backend/metadataArray,
        ./backend/memory_optimization_hints,
        ./backend/openmp,
        ./private/p_checks,
        ./private/p_accessors_macros_write,
        ./accessors, ./accessors_macros_syntax,
        ./data_structure, ./init_cpu,
        ./higher_order_applymap,
        ./higher_order_foldreduce,
        std/sequtils

func index_select*[T; Idx: byte or char or SomeInteger](t: Tensor[T], axis: int, indices: Tensor[Idx]): Tensor[T] {.noInit.} =
  ## Take elements from a tensor along an axis using the indices Tensor.
  ## This is equivalent to NumPy `take`.
  ## The result does not share the input storage, there are copies.
  ## The tensors containing the indices can be an integer, byte or char tensor.

  doAssert indices.shape.len == 1

  var select_shape = t.shape
  select_shape[axis] = indices.shape[0]
  result = newTensorUninit[T](select_shape)

  # TODO: optim for contiguous tensors
  # TODO: use OpenMP for tensors of non-ref/strings/seqs
  for i, index in enumerate(indices):
    var r_slice = result.atAxisIndex(axis, i)
    var t_slice = t.atAxisIndex(axis, int(index))
    r_slice.copyFrom(t_slice)

func index_select*[T; Idx: byte or char or SomeInteger](t: Tensor[T], axis: int, indices: openarray[Idx]): Tensor[T] {.noInit.} =
  ## Take elements from a tensor along an axis using the indices Tensor.
  ## This is equivalent to NumPy `take`.
  ## The result does not share the input storage, there are copies.
  ## The tensors containing the indices can be an integer, byte or char tensor.

  var select_shape = t.shape
  select_shape[axis] = indices.len
  result = newTensorUninit[T](select_shape)

  # TODO: optim for contiguous tensors
  # TODO: use OpenMP for tensors of non-ref/strings/seqs
  for i, index in indices:
    var r_slice = result.atAxisIndex(axis, i)
    var t_slice = t.atAxisIndex(axis, int(index))
    r_slice.copyFrom(t_slice)

func masked_select*[T](t: Tensor[T], mask: Tensor[bool]): Tensor[T] {.noInit.} =
  ## Take elements from a tensor according to the provided boolean mask
  ##
  ## Returns a **flattened** tensor which is the concatenation of values for which the mask is true.
  ##
  ## The result does not share input storage.
  check_elementwise(t, mask)

  # TODO: fold_inline should accept an accumType like fold_axis_inline
  var size = 0
  for val in mask:
    size += int(val)

  result = newTensorUninit[T](size)

  var idx = 0
  let dst{.restrict.} = result.dataArray
  for value, take in zip(t, mask):
    if take:
      dst[idx] = value
      inc idx
  assert idx == size

func masked_select*[T](t: Tensor[T], mask: openarray): Tensor[T] {.noInit.} =
  ## Take elements from a tensor according to the provided boolean mask
  ##
  ## The boolean mask must be
  ##   - an array or sequence of bools
  ##   - an array of arrays of bools,
  ##   - ...
  ##
  ## Returns a **flattened** tensor which is the concatenation of values for which the mask is true.
  ##
  ## The result does not share input storage.
  t.masked_select mask.toTensor()

func masked_axis_select*[T](t: Tensor[T], mask: Tensor[bool], axis: int): Tensor[T] {.noInit.} =
  ## Take elements from a tensor according to the provided boolean mask.
  ## The mask must be a 1D tensor and is applied along an axis, by default 0.
  ##
  ## The result will be the concatenation of values for which the mask is true.
  ##
  ## For example, for a 1D tensor `t`
  ## t.masked_select(t > 0) will return a tensor with
  ## only the positive values of t.
  ##
  ## The result does not share input storage.
  doAssert mask.shape.len == 1, "Mask must be a 1d tensor"

  # TODO: fold_inline should accept an accumType like fold_axis_inline
  var size = 0
  for val in mask:
    size += int(val)

  var shape = t.shape
  shape[axis] = size
  result = newTensorUninit[T](shape)

  # Track the current slice of the result tensor
  var dstSlice = shape.mapIt((0..<it)|1) # TODO avoid alloc

  dstSlice[axis].a = 0
  dstSlice[axis].b = 0

  for srcIndex, srcSlice in t.enumerateAxis(axis):
    if mask[srcIndex]:
      result.slicerMut(dstSlice, srcSlice)
      dstSlice[axis].a += 1
      dstSlice[axis].b = dstSlice[axis].a

  assert dstSlice[axis].a == size


func masked_axis_fill*[T](t: var Tensor[T], mask: Tensor[bool], axis: int, value: T or Tensor[T]) =
  ## Take a 1D boolean mask tensor with size equal to the `t.shape[axis]`
  ## The axis index that are set to true in the mask will be filled with `value`

  # TODO: proper check
  doAssert mask.shape.len == 1, "Mask must be a 1d tensor"

  # N-D tensor case, we iterate on t axis
  # We update the slice of t if mask is true.

  # Track the current slice of the result tensor
  var dstSlice = t.shape.mapIt((0..<it)|1) # TODO avoid alloc
  dstSlice[axis].a = 0
  dstSlice[axis].b = 0

  for fillIt in mask:
    if fillIt:
      t.slicerMut(dstSlice, value)
    dstSlice[axis].a += 1
    dstSlice[axis].b = dstSlice[axis].a


func masked_fill*[T](t: var Tensor[T], mask: Tensor[bool], value: T) =
  ## For the index of each element of t.
  ## Fill the elements at ``t[index]`` with the ``value``
  ## if their corresponding ``mask[index]`` is true.
  ## If not they are untouched.
  ##
  ## Example:
  ##
  ##   t.masked_fill(t > 0, -1)
  ##
  ## or alternatively:
  ##
  ##   t.masked_fill(t > 0): -1
  check_elementwise(t, mask)

  # Due to requiring unnecessary assigning `x` for a `false` mask
  # apply2_inline is a bit slower for very sparse mask.
  # As this is a critical operation, especially on dataframes, we use the lower level construct.
  #
  # t.apply2_inline(mask):
  #   if y:
  #     value
  #   else:
  #     x
  omp_parallel_blocks(block_offset, block_size, t.size):
    for tElem, maskElem in mzip(t, mask, block_offset, block_size):
      if maskElem:
        tElem = value


func masked_fill*[T](t: var Tensor[T], mask: openarray, value: T) =
  ## For the index of each element of t.
  ## Fill the elements at ``t[index]`` with the ``value``
  ## if their corresponding ``mask[index]`` is true.
  ## If not they are untouched.
  ##
  ## Example:
  ##
  ##   t.masked_fill(t > 0, -1)
  ##
  ## or alternatively:
  ##
  ##   t.masked_fill(t > 0): -1
  ##
  ## The boolean mask must be
  ##   - an array or sequence of bools
  ##   - an array of arrays of bools,
  ##   - ...
  ##
  t.masked_fill(mask.toTensor(), value)

func masked_fill_along_axis*[T](t: var Tensor[T], mask: Tensor[bool], axis: int, value: T) =
  ## Take a boolean mask tensor and
  ## for each slice of ``t`` along the ``axis``
  ## Set the slice elements to value if their mask is true

  # Normally we want to mutable axis iterator but Nim escape analysis prevents that
  # as it doesn't return a trivial type.
  # so we can't pass the slice to masked_fill / apply2_inline.
  #
  # for slice in t.axis(axis):
  #   slice.masked_fill(mask, value)

  when compileOption("boundChecks"):
    check_axis_index(t, axis, 0, t.shape[axis])
  var slice = t.atAxisIndex(axis, 0)

  for _ in 0 ..< t.shape[axis]:
    slice.masked_fill(mask, value)
    slice.offset += t.strides[axis]
