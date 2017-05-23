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

proc transpose*(t: Tensor): Tensor {.noSideEffect.}=
  ## Transpose a Tensor. For N-d Tensor with shape (0, 1, 2 ... n-1)
  ## the resulting tensor will have shape (n-1, ... 2, 1, 0)
  ## Data is copied as is and not modified.

  result.shape = t.shape.reversed
  result.strides = t.strides.reversed
  result.offset = t.offset
  result.data = t.data

proc asContiguous*[B,T](t: Tensor[B,T]): Tensor[B,T] {.noSideEffect.}=
  ## Transform a tensor with general striding to a Row major Tensor

  if t.isContiguous: return t

  result.shape = t.shape
  result.strides = shape_to_strides(t.shape)
  result.offset = 0
  result.data = newSeq[T](t.shape.product)

  var i = 0
  for val in t:
    result.data[i] = val
    inc i