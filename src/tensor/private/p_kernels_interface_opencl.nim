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

      dst_data[dst_real_idx] = A_data[A_real_idx] """ & op & """ B_data[B_real_idx];
    }
  }
  """


# Kernels are created just-in-time and incur some overhead
# unfortunately doing otherwise would require
# prebuilding binaries for each AMD, Nvidia, Intel, Qualcomm, ... OpenCL SDK and drivers
# Nvidia automatically caches OpenCL JIT compilation.
# For maximum performance you might need a similar scheme for your platform.

template genClInfixOp*( T: typedesc,
                        ctype: string,
                        procName: untyped,
                        cName: string,
                        cInfixOp: string,
                        exported: static[bool] = true): untyped =
  ## Generates an OpenCL kernel for an elementwise binary infix operation (like +, *, /, -)
  ## Input:
  ##   - The Nim type of the elements of the input tensors
  ##   - The equivalent C type
  ##   - The Nim identifier of the resulting proc
  ##   - The C kernel name (this only helps debugging the C code)
  ##   - The C operation (+, -, *, /)

  proc procName(a, b: ClTensor[T]): ClTensor[T] {.noInit.}=
    when compileOption("boundChecks"):
          check_elementwise(a,b)

    result = newClTensor[T](a.shape)

    let
      clKernel = gen_cl_apply3(cName, ctype, cInfixOp)
      program = clContext0.createAndBuild(clKernel, clDevice0)
      clProc = program.createKernel(cName)

      dst = layoutOnDevice result
      src_a = layoutOnDevice a
      src_b = layoutOnDevice b

    clProc.args(dst.rank, dst.len,
                dst.shape[], dst.strides[], dst.offset, dst.data.toClpointer,
                src_a.shape[], src_a.strides[], src_a.offset, src_a.data.toClpointer,
                src_b.shape[], src_b.strides[], src_b.offset, src_b.data.toClpointer
                )

    clQueue0.run(clProc, result.size)

  when exported:
    export procName
