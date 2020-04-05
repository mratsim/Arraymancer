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
        ./private/p_checks,
        ./private/p_accessors_macros_write,
        ./accessors,
        ./data_structure, ./init_cpu,
        ./higher_order_foldreduce,
        std/sequtils

func index_select*[T; Idx: byte or char or SomeNumber](t: Tensor[T], axis: int, indices: Tensor[Idx]): Tensor[T] =
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


func masked_select*[T](t: Tensor[T], mask: Tensor[bool], axis = 0): Tensor[T] =
  ## Take elements from a tensor according to the provided boolean mask.
  ## The mask must be a 1D tensor and is applied along an axis, by default 0.
  ##
  ## For example, for a 1D tensor `t`
  ## t.masked_select(t > 0) will return a tensor with
  ## only the positive values of t.
  ##
  ## The result does not share input storage.
  check_shape(mask, [1])

  let size = mask.reduce_inline():
    x += int(y)

  var shape = t.shape
  shape[axis] = size
  result = newTensorUninit[T](shape)

  if shape.len == 1:
    # 1D tensor case, we only need to iterate through ``t`` and ``mask`` and
    # copy ``t[i]`` if ``mask[i]`` is true
    check_elementwise(t, mask)
    var idx = 0
    let dst{.restrict.} = result.dataArray
    for value, take in zip(t, mask):
      if take:
        dst[idx] = value
        inc idx
    assert idx == size
  else:
    # N-D tensor case, we iterate on t axis
    # We copy the slice of t if mask is true.

    # Track the current slice of the result tensor
    var dstSlice = shape.mapIt((0..<it)|1) # TODO avoid alloc

    dstSlice[axis].a = 0
    dstSlice[axis].b = 0

    for srcIndex, srcSlice in t.enumerateAxis():
      if mask[srcIndex]:
        result.slicerMut(dstSlice, srcSlice)
        dstSlice[axis].a += 1
        dstSlice[axis].b = dstSlice[axis].a
        
    assert dstSlice[axis].a == size
