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

template opencl_getIndexOfElementID: string =
  """
  __global static inline int opencl_getIndexOfElementID(
    const int rank,
    __global const int * restrict const shape,
    __global const int * restrict const strides,
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
  """

template gen_cl_apply3*(kern_name, ctype, op: string): string =

  opencl_getIndexOfElementID() & """
  __kernel
  void """ & kern_name &
          """(const int rank,
              const int len,
              __global const int * restrict dst_shape,
              __global const int * restrict dst_strides,
              const int dst_offset,
              __global       """ & ctype & """ * restrict const dst_data,
              __global const int * restrict A_shape,
              __global const int * restrict A_strides,
              const int A_offset,
              __global const """ & ctype & """ * restrict const A_data,
              __global const int * restrict B_shape,
              __global const int * restrict B_strides,
              const int B_offset,
              __global const """ & ctype & """ * restrict const B_data)
  {
    // Grid-stride loop
    for (int elemID = get_global_id(0);
    elemID < len;
    elemID += get_global_size(0)) {
      const int dst_real_idx = opencl_getIndexOfElementID(rank, dst_shape, dst_strides, dst_offset, elemID);
      const int A_real_idx = opencl_getIndexOfElementID(rank, A_shape, A_strides, A_offset, elemID);
      const int B_real_idx = opencl_getIndexOfElementID(rank, B_shape, B_strides, B_offset, elemID);

      dst_data[elemID] = A_data[elemID] """ & op & """ B_data[elemID];
    }
  }
  """