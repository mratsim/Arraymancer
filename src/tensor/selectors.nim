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
        ./private/p_checks,
        ./accessors,
        ./data_structure, ./init_cpu

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
