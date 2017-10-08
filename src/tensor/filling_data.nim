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

proc copy_from*[T](dst: var Tensor[T], src: Tensor[T]) =
  ## Copy the data from a source Tensor. Both tensors must have the same number of elements
  ## but do not need to have the same shape.
  ## Data is copied without re-allocation.
  ## Warning ⚠
  ##   The destination tensor data will be overwritten. It however conserves its shape and strides.

  when compileOption("boundChecks"):
    check_size(dst, src)

  for x, val in mzip(dst, src):
    x = val