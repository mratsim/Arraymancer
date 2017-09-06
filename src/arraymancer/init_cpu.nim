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


template tensorCpu[T](out_shape: openarray[int], t: Tensor[T]): untyped =
  t.shape = @out_shape
  t.strides = shape_to_strides(t.shape)
  t.offset = 0

template toTensorCpu(s: typed): untyped =
  let shape = s.shape
  let data = toSeq(flatIter(s))

  when compileOption("boundChecks"): check_nested_elements(shape, data.len)

  var t: Tensor[type(data[0])]
  tensorCpu(shape, t)
  t.data = data
  return t


template randomTensorCpu[T](t: Tensor[T], shape: openarray[int], max_or_range: typed): untyped =
  tensor(shape, t)
  t.data = newSeqWith(t.shape.product, random(max_or_range))