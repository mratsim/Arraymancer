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
        std/[sequtils, locks]

# Indexed axis
# --------------------------------------------------------------------------------------------

proc index_select*[T; Idx: byte or char or SomeInteger](t: Tensor[T], axis: int, indices: Tensor[Idx]): Tensor[T] {.noinit.} =
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

proc index_select*[T; Idx: byte or char or SomeInteger](t: Tensor[T], axis: int, indices: openArray[Idx]): Tensor[T] {.noinit.} =
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

template index_fill_vector_body(): untyped {.dirty.} =
  if t.len == 0 or indices.len == 0:
    return
  if indices.len != values.len:
    raise newException(ValueError, "Cannot assign values to indices, because numbers mismatch: " &
      "# indices = " & $indices.len & ", # values = " & $values.len)
  when typeof(indices) isnot Tensor:
    template enumerate(arg): untyped {.gensym.} = pairs(arg)
  for i, index in enumerate(indices):
    var t_slice = t.atAxisIndex(axis, int(index))
    for old_val in t_slice.mitems():
      old_val = values[i]

# These are a bit terrible, but trying to overload `openArray[Idx] | Tensor[Idx]` for example doesn't seem to work
proc index_fill*[T; Idx: byte or char or SomeInteger](t: var Tensor[T], axis: int, indices: openArray[Idx], values: openArray[T]) =
  ## Replace elements of `t` indicated by their `indices` along `axis` with `value`
  ## This is equivalent to Numpy `put`.
  index_fill_vector_body()

proc index_fill*[T; Idx: byte or char or SomeInteger](t: var Tensor[T], axis: int, indices: Tensor[Idx], values: openArray[T]) =
  ## Replace elements of `t` indicated by their `indices` along `axis` with `value`
  ## This is equivalent to Numpy `put`.
  index_fill_vector_body()

proc index_fill*[T; Idx: byte or char or SomeInteger](t: var Tensor[T], axis: int, indices: openArray[Idx], values: Tensor[T]) =
  ## Replace elements of `t` indicated by their `indices` along `axis` with `value`
  ## This is equivalent to Numpy `put`.
  index_fill_vector_body()

proc index_fill*[T; Idx: byte or char or SomeInteger](t: var Tensor[T], axis: int, indices: Tensor[Idx], values: Tensor[T]) =
  ## Replace elements of `t` indicated by their `indices` along `axis` with `value`
  ## This is equivalent to Numpy `put`.
  index_fill_vector_body()

template index_fill_scalar_body(): untyped {.dirty.} =
  if t.size == 0 or indices.size == 0:
    return
  when typeof(indices) isnot Tensor:
    template enumerate(arg): untyped {.gensym.} = pairs(arg)
  for i, index in enumerate(indices):
    var t_slice = t.atAxisIndex(axis, int(index))
    for old_val in t_slice.mitems():
      old_val = value

proc index_fill*[T; Idx: byte or char or SomeInteger](t: var Tensor[T], axis: int, indices: Tensor[Idx], value: T) =
  ## Replace elements of `t` indicated by their `indices` along `axis` with `value`
  ## This is equivalent to Numpy `put`.
  index_fill_scalar_body()

proc index_fill*[T; Idx: byte or char or SomeInteger](t: var Tensor[T], axis: int, indices: openArray[Idx], value: T) =
  ## Replace elements of `t` indicated by their `indices` along `axis` with `value`
  ## This is equivalent to Numpy `put`.
  index_fill_scalar_body()

# Mask full tensor
# --------------------------------------------------------------------------------------------

proc masked_select*[T](t: Tensor[T], mask: Tensor[bool]): Tensor[T] {.noinit.} =
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
  let dst{.restrict.} = result.toUnsafeView
  for value, take in zip(t, mask):
    if take:
      dst[idx] = value
      inc idx
  assert idx == size

proc masked_select*[T](t: Tensor[T], mask: openArray): Tensor[T] {.noinit.} =
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
  ## For each element ``t[index]`` of the input tensor ``t`` with index ``index``,
  ## check if ``mask[index]`` is true. If so, fill it ``value``.
  ## Otherwise leave it untouched.
  ##
  ## Example:
  ##
  ##   t.masked_fill(t > 0, -1)
  ##
  ## or alternatively:
  ##
  ##   t.masked_fill(t > 0): -1
  ##
  ## In this version of this procedure the boolean mask is a ``Tensor[bool]``
  ## with the same size as the input tensor ``t``.

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

proc masked_fill*[T](t: var Tensor[T], mask: openArray, value: T) =
  ## For each element ``t[index]`` of the input tensor ``t`` with index ``index``,
  ## check if ``mask[index]`` is true. If so, fill it ``value``.
  ## Otherwise leave it untouched.
  ##
  ## Example:
  ##
  ##   t.masked_fill([true, false, true, true], -1)
  ##
  ## or alternatively:
  ##
  ##   t.masked_fill([true, false, true, true]): -1
  ##
  ## In this version of this procedure the boolean mask, which must have the
  ## same size as the input tensor ``t``, is an openArray of bools, i.e.:
  ##   - an array or sequence of bools
  ##   - an array of arrays of bools,
  ##   - ...
  if t.size == 0 or mask.len == 0:
    return
  t.masked_fill(mask.toTensor(), value)

template masked_fill_impl[T](t: var Tensor[T], mask: Tensor[bool], value: Tensor[T] | openArray[T]) =
  ## Implementation of masked_fill for both openArray and Tensor value
  ##
  ## It should have been possible to use a regular procedure to implement
  ## masked_fill both for openArrays an tensors. However, as for nim 2.0.2
  ## there are some limitations / bugs with implicit type conversions when
  ## applied to `or` typeclasses that contain openArrays. Because of that we've
  ## had to encapsulate the implementation of masked_fill in a template, and
  ## create 2 separate versions of the masked_fill procedure (one taking a
  ## tensor value and another taking an openArray value) which call this
  ## implementation template. Somehow this seems to work around the issue.

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

  # We need to protect against the case in which we run out of values to
  # fill the tensor with, which we cannot tell in advance without counting
  # the number of true values in the mask (which would be expensive)
  var lock: Lock
  initLock(lock)
  var too_few_values = false
  let value_size = value.len
  omp_parallel_blocks(block_offset, block_size, t.size):
    var n = block_offset
    for tElem, maskElem in mzip(t, mask, block_offset, block_size):
      if maskElem:
        if n >= value_size:
          withLock(lock):
            # The lock protection is technically unnecessary but it is good form
            too_few_values = true
          break
        tElem = value[n]
        inc n
  if too_few_values:
    let error_msg = "masked_fill error: the size of the value tensor (" & $value_size &
      ") is smaller than the number of true elements in the mask"
    when not(compileOption("mm", "arc") or compileOption("mm", "orc")):
      # Other memory management modes crash without showing the exception message
      echo error_msg
    raise newException(IndexDefect, error_msg)

proc masked_fill*[T](t: var Tensor[T], mask: Tensor[bool], value: Tensor[T]) =
  ## For each element ``t[index]`` of the input tensor ``t`` with index ``index``,
  ## check if ``mask[index]`` is true. If so fill it with the _next_
  ## element from the ``value`` tensor. Otherwise leave it untouched.
  ##
  ## Note that this does _not_ fill ``t[index]`` with ``value[index]``, but
  ## with the n-th element of ``value`` where n is the number of true elements
  ## in the mask before and including the index-th mask element.
  ## Because of this, the value tensor must have at least as many elements as
  ## the number of true elements in the mask. If that is not the case an
  ## IndexDefect exception will be raised at runtime. The ``value`` tensor
  ## can have even more values which will simply be ignored.
  ##
  ## Example:
  ##
  ##   t.masked_fill(t > 0, [3, 4, -1].toTensor)
  ##
  ## In this version of this procedure the boolean mask is a ``Tensor[bool]``
  ## with the same size as the input tensor ``t``.
  masked_fill_impl(t, mask, value)

proc masked_fill*[T](t: var Tensor[T], mask: Tensor[bool], value: openArray[T]) =
  ## Version of `masked_fill` that takes an openArray as the value
  ##
  ## For each element ``t[index]`` of the input tensor ``t`` with index ``index``,
  ## check if ``mask[index]`` is true. If so fill it with the _next_
  ## element from the ``value`` openArray. Otherwise leave it untouched.
  ##
  ## Note that this does _not_ fill ``t[index]`` with ``value[index]``, but
  ## with the n-th element of ``value`` where n is the number of true elements
  ## in the mask before and including the index-th mask element.
  ## Because of this, the value openArray must have at least as many elements as
  ## the number of true elements in the mask. If that is not the case an
  ## IndexDefect exception will be raised at runtime. The ``value`` tensor
  ## can have even more values which will simply be ignored.
  ##
  ## Example:
  ##
  ##   t.masked_fill(t > 0, [3, 4, -1])
  ##
  ## In this version of this procedure the boolean mask is a ``Tensor[bool]``
  ## with the same size as the input tensor ``t``.
  masked_fill_impl(t, mask, value)

proc masked_fill*[T](t: var Tensor[T], mask: openArray, value: Tensor[T]) =
  ## Version of masked_fill that takes an openArray[bool] as the mask
  ## and a tensor as the value
  if t.size == 0 or mask.len == 0:
    return
  masked_fill_impl(t, mask.toTensor(), value)

proc masked_fill*[T](t: var Tensor[T], mask: openArray, value: openArray[T]) =
  ## Version of masked_fill that takes an openArray[bool] as the mask
  ## and an openArray as the value
  if t.size == 0 or mask.len == 0:
    return
  masked_fill_impl(t, mask.toTensor(), value)

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

proc masked_axis_select*[T](t: Tensor[T], mask: Tensor[bool], axis: int): Tensor[T] {.noinit.} =
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

proc masked_axis_select*[T](t: Tensor[T], mask: openArray[bool], axis: int): Tensor[T] {.noinit.} =
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
  ## The axis indexes that are set to true in the mask will be filled with `value`
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
  ## The axis indexes that are set to true in the mask will be filled with `value`
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
