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

import  ../backend/cuda,
        ../backend/metadataArray,
        ../data_structure


template tensorCuda[T: SomeFloat](
  shape: typed,
  layout: OrderType = colMajor,
  result: var CudaTensor[T])=

  result.shape.copyFrom(shape)
  shape_to_strides(result.shape, layout, result.strides)
  result.offset = 0
  result.storage = newCudaStorage[T](result.size)

proc newCudaTensor*[T: SomeFloat](
  shape: varargs[int],
  layout: OrderType = colMajor): CudaTensor[T] {.noInit, noSideEffect.}=
  ## Internal proc
  ## Allocate a CudaTensor
  ## WARNING: The Cuda memory is not initialized to 0

  # TODO: default to RowMajor. Pending https://github.com/mratsim/Arraymancer/issues/22
  # As mentionned in design doc, an element-wise kernel will avoid relying on CuBLAS
  # for inplace operation that requires column major layout.

  tensorCuda(shape, layout, result)

proc newCudaTensor*[T: SomeFloat](
  shape: MetadataArray,
  layout: OrderType = colMajor): CudaTensor[T] {.noInit, noSideEffect.}=

  tensorCuda(shape, layout, result)
