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

const CUDA_HOF_TPB = 32 * 32 # TODO, benchmark and move that to cuda global config

{.emit: """
  template<typename T, typename Op>
  __global__ void cuda_apply2(const int rank,
                              const int len,
                              const int *a_shape,
                              const int *a_strides,
                              const int a_offset,
                              T *a_data,
                              const Op f,
                              const int *b_shape,
                              const int *b_strides,
                              const int b_offset,
                              T *b_data){

    for (int elemID = blockIdx.x * blockDim.x + threadIdx.x;
         elemID < len;
         elemID += blockDim.x * gridDim.x) {

      // ## we can't instantiate the variable outside the loop
      // ## each threads will store its own in parallel
      const int a_real_idx = cuda_getIndexOfElementID(
                               rank,
                               a_shape,
                               a_strides,
                               a_offset,
                               elemID);

      const int b_real_idx = cuda_getIndexOfElementID(
                               rank,
                               b_shape,
                               b_strides,
                               b_offset,
                               elemID);

      f(&a_data[a_real_idx], &b_data[b_real_idx]);
    }
  }
""".}