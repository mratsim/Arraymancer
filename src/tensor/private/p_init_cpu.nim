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

import  ../../private/nested_containers,
        ../backend/metadataArray,
        ../data_structure,
        ./checks,
        nimblas,
        sequtils

template tensorCpu*[T](out_shape: varargs[int], t: Tensor[T], layout: OrderType = rowMajor): untyped =
  t.shape = toMetadataArray(out_shape)
  t.strides = shape_to_strides(t.shape, layout)
  t.offset = 0

template tensorCpu*[T](out_shape: MetadataArray, t: Tensor[T], layout: OrderType = rowMajor): untyped =
  t.shape = out_shape
  t.strides = shape_to_strides(t.shape, layout)
  t.offset = 0

template toTensorCpu*(s: typed): untyped =
  let shape = s.shape
  let data = toSeq(flatIter(s))

  when compileOption("boundChecks"):
    check_nested_elements(shape, data.len)

  var t: Tensor[type(data[0])]
  tensorCpu(shape, t)
  t.data = data
  return t

proc newSeqUninit*[T](len: Natural): seq[T] {.noSideEffect, inline.} =
  result = newSeqOfCap[T](len)
  result.setLen(len)