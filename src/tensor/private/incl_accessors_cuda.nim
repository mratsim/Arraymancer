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

# For element-wise operations, instead of a sequential iterator like for CPU,
# it will be faster to have many threads compute the index -> offset and update
# the data at this offset.
#
# For this we need:
#   - to store strides and offset on the cuda device to avoid copies
#   - a way to convert element #10 of the tensor to the real offset (column major),
#     the kernels won't use tensor[2,5] as an index


# proc getIndexOfElementID[T](t: Tensor[T], element_id: int): int {.noSideEffect,used.} =
#   ## Convert "Give me element 10" to the real index/memory offset.
#   ## Reference Nim CPU version
#   ## This is not meant to be used on serial architecture due to the division overhead.
#   ## On GPU however it will allow threads to address the real memory addresses independantly.
#
#   when compileOption("boundChecks"):
#     assert element_id < t.size
#
#   result = t.offset
#   var currentOffset = element_id
#   var dimIdx: int
#
#   for k in countdown(t.rank - 1,0):
#     ## hopefully the compiler doesn't do division twice ...
#     dimIdx = currentOffset mod t.shape[k]
#     currentOffset = currentOffset div t.shape[k]
#
#     # cf atIndex proc to compute real_idx
#     result += dimIdx * t.strides[k]

# Note we don't bound-checks the CUDA implementation
{.emit:["""
  static inline __device__ int cuda_getIndexOfElementID(
    const int rank,
    const int * __restrict__ shape,
    const int * __restrict__ strides,
    const int offset,
    const int element_id) {

    int real_idx = offset;
    int currentOffset = element_id;
    int dimIdx = 0;

    for (int k = rank - 1; k >= 0; --k) {
      dimIdx = currentOffset % shape[k];
      currentOffset /= shape[k];

      real_idx += dimIdx * strides[k];
    }

    return real_idx;
  }
  """].}