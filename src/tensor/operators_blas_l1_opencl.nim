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


import  ./backend/opencl_backend,
        ./private/p_kernels_interface_opencl,
        ./private/p_init_opencl,
        ./private/p_checks,
        ./data_structure

# Kernels are created just-in-time and incur some overhead
# unfortunately doing otherwise would require
# prebuilding binaries for each AMD, Nvidia, Intel, Qualcomm, ... OpenCL SDK and drivers
# Nvidia automatically caches OpenCL JIT compilation.
# For maximum performance you might need a similar scheme for your platform.

template genAdd(T: typedesc, ctype: string): untyped =
  proc `+`*(a,b: ClTensor[T]): ClTensor[T] {.noInit.}=
    ## ClTensor addition

    when compileOption("boundChecks"):
      check_elementwise(a,b)

    result = newClTensor[T](a.shape)

    let
      ocl_addKernel = gen_ocl_apply3("AddKernel", ctype, "+")
      program = clContext0.createAndBuild(ocl_addKernel, clDevice0)
      opencl_add = program.createKernel("AddKernel")

      dst = layoutOnDevice result
      src_a = layoutOnDevice a
      src_b = layoutOnDevice b

    opencl_add.args(dst.rank, dst.len,
                        dst.shape[], dst.strides[], dst.offset, dst.data.toClpointer,
                        src_a.shape[], src_a.strides[], src_a.offset, src_a.data.toClpointer,
                        src_b.shape[], src_b.strides[], src_b.offset, src_b.data.toClpointer
                        )

    clQueue0.run(opencl_add, result.size)

genAdd(float32, "float")
genAdd(float64, "double")