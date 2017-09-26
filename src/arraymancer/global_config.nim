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




# This configures the maximum number of dimensions supported by Arraymancer
# It should improve performance on Cuda and for iterator by storing temporary shape/strides
# that will be used extensively in the loop on the stack.
# For now this is only partly implemented and only on Cuda temporary shape/strides arrays.
const MAXRANK = 8 # 8 because it's a nice number, more is possible upon request.


const CUDA_HOF_TPB: cint = 32 * 32 # TODO, benchmark and move that to cuda global config
                                   # Pascal GTX 1070+ have 1024 threads max
const CUDA_HOF_BPG: cint = 256     # should be (grid-stride+threadsPerBlock-1) div threadsPerBlock ?
                                   # From https://devblogs.nvidia.com/parallelforall/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/
                                   # Lower allows threads re-use and limit overhead of thread creation/destruction


const OMP_FOR_THRESHOLD = 1000    # Tensor number of elements threshold before using OpenMP multithreading

# Full procesor optimization (AVX, AVX2, ARM neon, ... if applicable)
when defined(native):
  {.passC: "-march=native".}

# Note: Following https://github.com/mratsim/Arraymancer/issues/61 and
# https://github.com/mratsim/Arraymancer/issues/43
# Arraymancer export '_' for slicing (type is SteppedSlice)
# '_' is configured in accessors_slicer