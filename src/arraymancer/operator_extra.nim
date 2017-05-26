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

proc `|*|`*[T: SomeNumber](a, b: Tensor[Cpu,T]): Tensor[Cpu,T] {.noSideEffect.} =
  ## Element-wise multiplication (hadamard product)
  ## TODO: find a good symbol
  when compileOption("boundChecks"): check_add(a,b)

  result.shape = a.shape
  result.strides = shape_to_strides(a.shape)
  result.data = newSeq[T](a.shape.product)
  result.offset = 0

  ## TODO use mitems instead of result.data[i] cf profiling
  for i, ai, bi in enumerate_zip(a.values, b.values):
    result.data[i] = ai * bi