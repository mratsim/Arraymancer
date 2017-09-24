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

template toTensorReshapeT(oa: typed, shape: varargs[int]): untyped = 
  let data = toSeq(flatIter(oa))
  let seq_shape = @shape

  when compileOption("boundChecks"): check_nested_elements(seq_shape, data.len)

  var t: Tensor[type(data[0])]
  tensorCpu(seq_shape, t)
  shallowCopy(t.data, data)
  return t

proc toTensorReshape(oa: string, shape: varargs[int]): auto {.noSideEffect.}=
  ## Fuse toTensor and reshape in one operation.
  ##
  ## Deal specifically with strings/seq[char]

  toTensorReshapeT(oa, shape)

proc toTensorReshape(oa: openarray, shape: varargs[int], dummy_bugfix: static[int] = 0): auto {.noSideEffect.}=
  ## Fuse toTensor and reshape in one operation
  ##
  ## Nim >0.17 needed as "static[int] = 0" is not working in Nim 0.17
  ## Dummy_bugfix param is necessary due to: https://github.com/nim-lang/Nim/issues/6343
  # TODO: remove 'dummy_bugfix'
  toTensorReshapeT(oa, shape)

template rewriteToTensorReshape*{reshape(toTensor(oa, dummy_bugfix), shape)}(
  oa: openarray,
  shape: varargs[int],
  dummy_bugfix: static[int]): auto =
  ## Fuse ``sequence.toTensor.reshape(new_shape)`` into a single operation.
  ##
  ## Operation fusion leverage the Nim compiler and should not be called explicitly.
  toTensorReshape(oa, shape, dummy_bugfix)