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


# Note: Maximum number of threads per block is
# 1024 on Pascal GPU, i.e. 32 warps of 32 threads


# Important CUDA optimization
# To loop over each element of an array with arbitrary length
# use grid-strides for loop: https://devblogs.nvidia.com/parallelforall/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/
#
# Avoid branching in the same warp (32 threads), otherwise it reverts to serial execution.
# "idx < length" can be converted to "idx = max( idx, 0); idx = min( idx, length);"
# for example. (Beware of aliasing)

# TODO, use an on-device struct to store, shape, strides, offset

const CUDA_HOF_TPB: cint = 32 * 32 # TODO, benchmark and move that to cuda global config
                                   # Pascal GTX 1070+ have 1024 threads max
const CUDA_HOF_BPG: cint = 256     # should be (grid-stride+threadsPerBlock-1) div threadsPerBlock ?
                                   # From https://devblogs.nvidia.com/parallelforall/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/
                                   # Lower allows threads re-use and limit overhead of thread creation/destruction


{.emit: """
  template<typename T, typename Op>
  __global__ void cuda_apply2(const int rank,
                              const int len,
                              const int *  __restrict__ dst_shape,
                              const int *  __restrict__ dst_strides,
                              const int dst_offset,
                              T * __restrict__ dst_data,
                              Op f,
                              const int *  __restrict__ src_shape,
                              const int *  __restrict__ src_strides,
                              const int src_offset,
                              const T * __restrict__ src_data){

    for (int elemID = blockIdx.x * blockDim.x + threadIdx.x;
         elemID < len;
         elemID += blockDim.x * gridDim.x) {

      // ## we can't instantiate the variable outside the loop
      // ## each threads will store its own in parallel
      const int dst_real_idx = cuda_getIndexOfElementID(
                               rank,
                               dst_shape,
                               dst_strides,
                               dst_offset,
                               elemID);

      const int src_real_idx = cuda_getIndexOfElementID(
                               rank,
                               src_shape,
                               src_strides,
                               src_offset,
                               elemID);

      f(&dst_data[dst_real_idx], &src_data[src_real_idx]);
    }
  }
""".}