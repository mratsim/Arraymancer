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

import  ../backend/opencl_backend,
        ../data_structure


template tensorOpenCL[T: SomeFloat](
  shape: typed,
  layout: OrderType = rowMajor,
  result: var ClTensor[T])=

  result.shape.copyFrom(shape)
  shape_to_strides(result.shape, layout, result.strides)
  result.offset = 0
  result.storage = newClStorage[T](result.size)

proc newClTensor*[T: SomeFloat](
  shape: varargs[int],
  layout: OrderType = rowMajor): ClTensor[T] {.noinit.}=
  ## Internal proc
  ## Allocate a ClTensor
  ## WARNING: The OpenCL memory is not initialized to 0

  tensorOpenCL(shape, layout, result)

proc newClTensor*[T: SomeFloat](
  shape: Metadata,
  layout: OrderType = rowMajor): ClTensor[T] {.noinit.}=

  tensorOpenCL(shape, layout, result)
