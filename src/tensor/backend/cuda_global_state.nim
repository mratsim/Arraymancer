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


# ###################################################
# Global Cuda and CuBLAS state

# CuBLAS stream for parallel async processing on GPU
# Computations/Memcpy on different streams are done in simultaneously
# Streams are also necessary for async Cuda procs like cudaMemcpyAsync
var defaultStream: cublas_api.cudaStream_t
check cudaStreamCreate(addr defaultStream)

# CuBLAS handle
# Note: it prevents {.noSideEffect.} in all CuBLAS proc :/
var defaultHandle: cublasHandle_t
check cublasCreate(addr defaultHandle)

proc cudaRelease() {.noconv.}=
  # Release all cuda resources
  check cublasDestroy(defaultHandle)
  check cudaStreamDestroy(defaultStream)

  when defined(debug):
    echo "CUDA and CuBLAS resources successfully released"

addQuitProc(cudaRelease)
