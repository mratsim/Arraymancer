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

import nimcuda/[nimcuda, cuda_runtime_api, cublas_v2, cublas_api]

# ###################################################
# Global Cuda and CuBLAS state

proc initCudaStream(): cudaStream_t =
  ## CuBLAS stream for parallel async processing on GPU
  ## Computations/Memcpy on different streams are done in simultaneously
  ## Streams are also necessary for async Cuda procs like cudaMemcpyAsync
  check cudaStreamCreate(addr result)

proc initCublasHandle(): cublasHandle_t =
  check cublasCreate(addr result)

{.experimental.}
proc `=destroy`(c: cublasHandle_t) =
  check cublasDestroy(c)

proc `=destroy`(c: cudaStream_t) =
  check cudaStreamDestroy(c)

let cudaStream0* = initCudaStream()
let cublasHandle0*  = initCublasHandle()