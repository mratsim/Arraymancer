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

import  ./private/p_init_cuda,
        ./private/p_kernels_interface_cuda,
        ./data_structure

include ./private/incl_accessors_cuda,
        ./private/incl_higher_order_cuda,
        ./private/incl_kernels_cuda


cuda_assign_glue("cuda_clone", "CopyOp", cuda_clone)

proc clone*[T](t: CudaTensor[T]): CudaTensor[T] {.noInit, noSideEffect.}=
  ## Clone (deep copy) a CudaTensor.
  ## Copy will not share its data with the original.

  result = newCudaTensor[T](t.shape, colMajor) # TODO: change to rowMajor

  cuda_assign_call(cuda_clone, result, t)