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

import  ./backend/memory_optimization_hints,
        ./backend/openmp,
        ./private/p_checks,
        ./private/p_accessors_macros_write,
        ./private/p_empty_tensors,
        ./accessors, ./accessors_macros_syntax,
        ./data_structure, ./init_cpu,
        ./higher_order_applymap,
        ./higher_order_foldreduce,
        std/sequtils

# Indexed axis
# --------------------------------------------------------------------------------------------

proc index_select*[T; Idx: byte or char or SomeInteger](t: Tensor[T], axis: int, indices: Tensor[Idx]): Tensor[T] {.noInit.} =
  ## Take elements from a tensor along an axis using the indices Tensor.
  ## This is equivalent to NumPy `take`.
  ## The result does not share the input storage, there are copies.
  ## The tensors containing the indices can be an integer, byte or char tensor.
  returnEmptyIfEmpty(t, indices)
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

proc index_select*[T; Idx: byte or char or SomeInteger](t: Tensor[T], axis: int, indices: openarray[Idx]): Tensor[T] {.noInit.} =
  ## Take elements from a tensor along an axis using the indices Tensor.
  ## This is equivalent to NumPy `take`.
  ## The result does not share the input storage, there are copies.
  ## The tensors containing the indices can be an integer, byte or char tensor.
  returnEmptyIfEmpty(t)
  var select_shape = t.shape
  select_shape[axis] = indices.len
  result = newTensorUninit[T](select_shape)

  # TODO: optim for contiguous tensors
  # TODO: use OpenMP for tensors of non-ref/strings/seqs
  for i, index in indices:
    var r_slice = result.atAxisIndex(axis, i)
    var t_slice = t.atAxisIndex(axis, int(index))
    r_slice.copyFrom(t_slice)

proc index_fill*[T; Idx: byte or char or SomeInteger](t: var Tensor[T], axis: int, indices: Tensor[Idx], value: T) =
  ## Replace elements of `t` indicated by their `indices` along `axis` with `value`
  ## This is equivalent to Numpy `put`.
  if t.size == 0 or indices.size == 0:
    return
  for i, index in enumerate(indices):
    var t_slice = t.atAxisIndex(axis, int(index))
    for old_val in t_slice.mitems():
      old_val = value

proc index_fill*[T; Idx: byte or char or SomeInteger](t: var Tensor[T], axis: int, indices: openarray[Idx], value: T) =
  ## Replace elements of `t` indicated by their `indices` along `axis` with `value`
  ## This is equivalent to Numpy `put`.
  if t.size == 0 or indices.len == 0:
    return
  for i, index in indices:
    var t_slice = t.atAxisIndex(axis, int(index))
    for old_val in t_slice.mitems():
      old_val = value

# Mask full tensor
# --------------------------------------------------------------------------------------------

proc masked_select*[T](t: Tensor[T], mask: Tensor[bool]): Tensor[T] {.noInit.} =
  ## Take elements from a tensor according to the provided boolean mask
  ##
  ## Returns a **flattened** tensor which is the concatenation of values for which the mask is true.
  ##
  ## The result does not share input storage.
  returnEmptyIfEmpty(t, mask)
  check_elementwise(t, mask)

  # TODO: fold_inline should accept an accumType like fold_axis_inline
  var size = 0
  for val in mask:
    size += int(val)

  if size == 0:
    return typeof(result)() # initialized empty

  result = newTensorUninit[T](size)
  withMemoryOptimHints()

  var idx = 0
  let dst{.restrict.} = result.dataArray
  for value, take in zip(t, mask):
    if take:
      dst[idx] = value
      inc idx
  assert idx == size

proc masked_select*[T](t: Tensor[T], mask: openarray): Tensor[T] {.noInit.} =
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

proc masked_fill*[T](t: var Tensor[T], mask: Tensor[bool], value: T) =
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
  if t.size == 0 or mask.size == 0:
    return
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


proc masked_fill*[T](t: var Tensor[T], mask: openarray, value: T) =
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
  if t.size == 0 or mask.len == 0:
    return
  t.masked_fill(mask.toTensor(), value)

# Mask axis
# --------------------------------------------------------------------------------------------

template masked_axis_select_impl[T](result: var Tensor[T], t: Tensor[T], mask: Tensor[bool] or openArray[bool], axis: int) =
  ## Indirection because Nim proc can't type match "Tensor[bool] or openArray[bool]" with an array[N, bool]
  when mask is Tensor:
    doAssert mask.shape.len == 1, "Mask must be a 1d tensor"
    doAssert t.shape[axis] == mask.shape[0], "The mask length doesn't match the axis length."
  else:
    doAssert t.shape[axis] == mask.len, "The mask length doesn't match the axis length."

  # TODO: fold_inline should accept an accumType like fold_axis_inline
  var size = 0
  for val in mask:
    size += int(val)

  if size == 0:
    return typeof(result)() # initialized empty

  var shape = t.shape
  shape[axis] = size
  result = newTensorUninit[T](shape)

  # Track the current slice of the result tensor
  var dstSlice = mapIt(shape, (0..<it)|1) # TODO avoid alloc

  dstSlice[axis].a = 0
  dstSlice[axis].b = 0

  for srcIndex, srcSlice in t.enumerateAxis(axis):
    if mask[srcIndex]:
      result.slicerMut(dstSlice, srcSlice)
      dstSlice[axis].a += 1
      dstSlice[axis].b = dstSlice[axis].a

  assert dstSlice[axis].a == size, "Found " & $size &
    " true values in mask, but selected " & $dstSlice[axis].a &
    " values on that axis"

proc masked_axis_select*[T](t: Tensor[T], mask: Tensor[bool], axis: int): Tensor[T] {.noInit.} =
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
  returnEmptyIfEmpty(t, mask)
  let mask = mask.squeeze() # make 1D if coming from unreduced axis aggregation like sum
  masked_axis_select_impl(result, t, mask, axis)

proc masked_axis_select*[T](t: Tensor[T], mask: openArray[bool], axis: int): Tensor[T] {.noInit.} =
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
  returnEmptyIfEmpty(t, mask)
  masked_axis_select_impl(result, t, mask, axis)

template masked_axis_fill_impl[T](t: var Tensor[T], mask: Tensor[bool] or openArray[bool], axis: int, value: T or Tensor[T]) =
  ## Indirection because Nim proc can't type match "Tensor[bool] or openArray[bool]" with an array[N, bool]
  # TODO: proper check
  when mask is Tensor:
    doAssert mask.shape.len == 1, "Mask must be a 1d tensor"
    doAssert t.shape[axis] == mask.shape[0], "The mask length doesn't match the axis length."
  else:
    doAssert t.shape[axis] == mask.len, "The mask length doesn't match the axis length."

  # N-D tensor case, we iterate on t axis
  # We update the slice of t if mask is true.

  # Track the current slice of the result tensor
  var dstSlice = mapIt(t.shape, (0..<it)|1) # TODO avoid alloc
  dstSlice[axis].a = 0
  dstSlice[axis].b = 0

  for fillIt in mask:
    if fillIt:
      t.slicerMut(dstSlice, value)
    dstSlice[axis].a += 1
    dstSlice[axis].b = dstSlice[axis].a

proc masked_axis_fill*[T](t: var Tensor[T], mask: Tensor[bool], axis: int, value: T or Tensor[T]) =
  ## Take a 1D boolean mask tensor with size equal to the `t.shape[axis]`
  ## The axis index that are set to true in the mask will be filled with `value`
  ##
  ## Limitation:
  ##   If value is a Tensor, only filling via broadcastable tensors is supported at the moment
  ##   for example if filling axis of a tensor `t` of shape [4, 3] the corresponding shapes are valid
  ##     [4, 3].masked_axis_fill(mask = [1, 3], axis = 1, value = [4, 1])
  ##
  ##   with values
  ##     t = [[ 4, 99,  2],
  ##          [ 3,  4, 99],
  ##          [ 1,  8,  7],
  ##          [ 8,  6,  8]].toTensor()
  ##     mask = [false, true, true]
  ##     value = [[10],
  ##              [20],
  ##              [30],
  ##              [40]].toTensor()
  ##
  ##     result = [[  4, 10, 10],
  ##               [  3, 20, 20],
  ##               [  1, 30, 30],
  ##               [  8, 40, 40]].toTensor()
  # TODO: support filling with a multidimensional tensor
  if t.size == 0 or mask.size == 0:
    return
  when value is Tensor:
    if value.size == 0:
      return
  let mask = mask.squeeze() # make 1D if coming from unreduced axis aggregation like sum
                            # TODO: squeeze exactly depending on axis to prevent accepting invalid values
  masked_axis_fill_impl(t, mask, axis, value)

proc masked_axis_fill*[T](t: var Tensor[T], mask: openArray[bool], axis: int, value: T or Tensor[T]) =
  ## Take a 1D boolean mask tensor with size equal to the `t.shape[axis]`
  ## The axis index that are set to true in the mask will be filled with `value`
  ##
  ## Limitation:
  ##   If value is a Tensor, only filling via broadcastable tensors is supported at the moment
  ##   for example if filling axis of a tensor `t` of shape [4, 3] the corresponding shapes are valid
  ##     [4, 3].masked_axis_fill(mask = [1, 3], axis = 1, value = [4, 1])
  ##
  ##   with values
  ##     t = [[ 4, 99,  2],
  ##          [ 3,  4, 99],
  ##          [ 1,  8,  7],
  ##          [ 8,  6,  8]].toTensor()
  ##     mask = [false, true, true]
  ##     value = [[10],
  ##              [20],
  ##              [30],
  ##              [40]].toTensor()
  ##
  ##     result = [[  4, 10, 10],
  ##               [  3, 20, 20],
  ##               [  1, 30, 30],
  ##               [  8, 40, 40]].toTensor()
  # TODO: support filling with a multidimensional tensor
  if t.size == 0 or mask.len == 0:
    return
  when value is Tensor:
    if value.size == 0:
      return
  masked_axis_fill_impl(t, mask, axis, value)

# Apply N-D mask along an axis
# --------------------------------------------------------------------------------------------

proc masked_fill_along_axis*[T](t: var Tensor[T], mask: Tensor[bool], axis: int, value: T) =
  ## Take a boolean mask tensor and
  ## for each slice of ``t`` along the ``axis``
  ## Set the slice elements to value if their mask is true

  # Normally we want to mutable axis iterator but Nim escape analysis prevents that
  # as it doesn't return a trivial type.
  # so we can't pass the slice to masked_fill / apply2_inline.
  #
  # for slice in t.axis(axis):
  #   slice.masked_fill(mask, value)
  if t.size == 0 or mask.size == 0:
    return
  when compileOption("boundChecks"):
    check_axis_index(t, axis, 0, t.shape[axis])
  var slice = t.atAxisIndex(axis, 0)

  for _ in 0 ..< t.shape[axis]:
    slice.masked_fill(mask, value)
    slice.offset += t.strides[axis]
