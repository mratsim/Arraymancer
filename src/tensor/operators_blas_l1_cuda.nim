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

import  ./backend/cublas,
        ./private/p_kernels_interface_cuda,
        ./private/p_init_cuda,
        ./private/p_checks,
        ./data_structure

include ./private/incl_accessors_cuda,
        ./private/incl_higher_order_cuda,
        ./private/incl_kernels_cuda

# ####################################################################
# BLAS Level 1 (Vector dot product, Addition, Scalar to Vector/Matrix)

proc dot*[T: SomeReal](a, b: CudaTensor[T]): T {.inline.}=
  ## Vector to Vector dot (scalar) product
  when compileOption("boundChecks"):
    check_dot_prod(a,b)
  cublas_dot( a.shape[0],
              a.get_data_ptr, a.strides[0],
              b.get_data_ptr, b.strides[0],
              addr result)

cuda_assign_glue("cuda_mAdd", "mAddOp", cuda_mAdd)

proc `+=`*[T: SomeReal](a: var CudaTensor[T], b: CudaTensor[T]) =
  ## CudaTensor in-place addition

  when compileOption("boundChecks"):
    check_elementwise(a,b)

  cuda_assign_call(cuda_mAdd, a, b)

  # TODO: if a and b share the same location, TEST

cuda_binary_glue("cuda_Add", "AddOp", cuda_Add)

proc `+`*[T: SomeReal](a,b: CudaTensor[T]): CudaTensor[T] {.noInit.}=
  ## CudaTensor addition

  when compileOption("boundChecks"):
    check_elementwise(a,b)

  result = newCudaTensor[T](a.shape)
  cuda_binary_call(cuda_Add, result, a, b)

cuda_assign_glue("cuda_mSub", "mSubOp", cuda_mSub)

proc `-=`*[T: SomeReal](a: var CudaTensor[T], b: CudaTensor[T]) =
  ## CudaTensor in-place substraction

  when compileOption("boundChecks"):
    check_elementwise(a,b)

  cuda_assign_call(cuda_mSub, a, b)

  # TODO: if a and b share the same location, TEST


cuda_binary_glue("cuda_Sub", "SubOp", cuda_Sub)

proc `-`*[T: SomeReal](a,b: CudaTensor[T]): CudaTensor[T] {.noInit.} =
  ## CudaTensor substraction

  when compileOption("boundChecks"):
    check_elementwise(a,b)

  result = newCudaTensor[T](a.shape)
  cuda_binary_call(cuda_Sub, result, a, b)

proc `*=`*[T:SomeReal](t: var CudaTensor[T]; a: T) {.inline.}=
  ## CudaTensor inplace multiplication by a scalar

  # We multiply all elements of the CudaTensor regardless of shape/strides
  # So this operation can be applied to tensors of all ranks.
  # Hence we use the whole allocated length and a stride of 1
  cublas_scal(t.data.len, a, t.get_data_ptr, 1)

proc `*`*[T:SomeReal](a: T, t: CudaTensor[T]): CudaTensor[T] {.noInit, inline.}=
  ## CudaTensor multiplication by a scalar

  # TODO replace by a custom kernel
  # Instead of a full clone we keep only the useful which is advantageous if t was a slice
  # It also makes it contiguous
  result = t.clone()
  result *= a

proc `*`*[T:SomeReal](t: CudaTensor[T], a: T): CudaTensor[T] {.noInit, inline.}=
  ## CudaTensor multiplication by a scalar
  a * t

proc `/=`*[T:SomeReal](t: var CudaTensor[T]; a: T) {.inline.}=
  ## CudaTensor in-place division by a scalar
  t *= (1/a)

cuda_rscal_glue("cuda_rscalDiv","RscalDiv", cuda_rscalDiv)

proc `/`*[T: SomeReal](t: CudaTensor[T], val: T): CudaTensor[T] {.noInit.} =
  ## CudaTensor division by a scalar
  result = newCudaTensor[T](t.shape)
  cuda_rscal_call(cuda_rscalDiv, result, t, val)