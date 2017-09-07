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


# Note: the following handle prevent {.noSideEffect.} in all CuBLAS proc :/
var defaultHandle: cublasHandle_t
check cublasCreate_v2(addr defaultHandle)

# #####################################################
# Redefinition of imported CUDA proc with standard name
proc dot*(handle: cublasHandle_t; n: int32; x: ptr float32; incx: int32;
                     y: ptr float32; incy: int32; output: ptr float32): cublasStatus_t {.inline.} =
  cublasSdot_v2(handle, n, x, incx, y, incy, output)

proc dot*(handle: cublasHandle_t; n: int32; x: ptr float64; incx: int32;
                     y: ptr float64; incy: int32; output: ptr float64): cublasStatus_t {.inline.} =
  cublasDdot_v2(handle, n, x, incx, y, incy, output)


# ####################################################################
# BLAS Level 1 (Vector dot product, Addition, Scalar to Vector/Matrix)

proc `.*`*[T: SomeReal](a, b: CudaTensor[T]): T =
  ## Vector to Vector dot (scalar) product
  when compileOption("boundChecks"): check_dot_prod(a,b)
  check dot(defaultHandle,
            a.shape[0].cint,
            a.get_data_ptr, a.strides[0].cint,
            b.get_data_ptr, b.strides[0].cint,
            addr result)