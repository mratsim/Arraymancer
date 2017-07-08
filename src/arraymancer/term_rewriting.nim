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

template toTensorReshapeT(oa: typed, B: static[Backend], shape: varargs[int]): untyped = 
  let data = toSeq(flatIter(oa))
  let seq_shape = @shape

  when compileOption("boundChecks"): check_nested_elements(seq_shape, data.len)

  result = emptyTensor(seq_shape, type(data[0]), B)
  result.data = data

proc toTensorReshape(oa: string, B: static[Backend], shape: varargs[int]): auto =
  ## Fuse toTensor and reshape in one operation
  ## Deal specifically with strings/seq[char]

  toTensorReshapeT(oa, B, shape)

proc toTensorReshape(oa: openarray, B: static[Backend], shape: varargs[int]): auto =
  ## Fuse toTensor and reshape in one operation

  toTensorReshapeT(oa, B, shape)

template rewriteToTensorReshape*{reshape(toTensor(oa, B), shape)}(
  oa: openarray,
  B: static[Backend],
  shape: varargs[int]): auto =
  ## Fuse ``sequence.toTensor(Backend).reshape(new_shape)`` into a single operation.
  toTensorReshape(oa, B, shape)