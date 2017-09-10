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

# ####################################################################
# Helper proc

proc cudaVV_A_eq_A_p_bB[T: SomeReal](
  a: var CudaTensor[T], beta: T, b: CudaTensor[T]) {.inline.}=
  # Vector: A = A + beta B

  cublas_axpy(a.shape[0],
              beta,
              b.get_data_ptr, b.strides[0],
              a.get_data_ptr, a.strides[0])

proc cudaVV_C_eq_A_p_bB[T: SomeReal]( a: CudaTensor[T],
                                      beta: T, b: CudaTensor[T],
                                      result: var CudaTensor[T]) {.inline.}=
  # Vector: C = A + beta B
  result = newCudaTensor[T](a.shape)

  cublas_copy(a.len, a.get_data_ptr, a.strides[0],
              result.get_data_ptr, result.strides[0])

  cudaVV_A_eq_A_p_bB(result, beta, b)

proc cudaMM_A_eq_aA_p_bB[T: SomeReal](
  alpha: T, a: var CudaTensor[T],
  beta: T, b: CudaTensor[T]) =
  # Matrix: A = alpha A + beta B

  # TODO: remove this contiguous layout constraint (via conversion or custom kernel)
  if not (isContiguous(a) and isContiguous(b)):
    raise newException(ValueError, "NotImplemented: for now both tensors should be contiguous")

  if not is_F_contiguous(a):
    raise newException(ValueError, "NotImplemented: the modified tensor must have a column-major layout")

  let
    b_is_colMajor = b.is_F_contiguous

    transpose_B = if b_is_colMajor: CUBLAS_OP_N
                  else: CUBLAS_OP_T

    ld_B =  if b_is_colMajor: b.strides[1]
            else: b.strides[0]

  cublas_geam(CUBLAS_OP_N, transpose_B,
              a.shape[0], a.shape[1],
              alpha,
              a.get_data_ptr, a.strides[1],
              beta,
              b.get_data_ptr, ld_B,
              a.get_data_ptr, a.strides[1])
  # In column-majour layout a.shape[0] == a.strides[1]

proc cudaMM_C_eq_aA_p_aB[T: SomeReal](alpha: T, a: CudaTensor[T],
                                          beta: T, b: CudaTensor[T],
                                          result: var CudaTensor[T]) =
  # TODO: remove this contiguous layout constraint (via conversion or custom kernel)
  if not (isContiguous(a) and isContiguous(b)):
    raise newException(ValueError, "NotImplemented: for now both tensors should be contiguous")

  result = newCudaTensor[T](a.shape) # result is colMajor

  let
    a_is_colMajor = a.is_F_contiguous
    b_is_colMajor = b.is_F_contiguous

    transpose_A = if a_is_colMajor: CUBLAS_OP_N
                  else: CUBLAS_OP_T
    ld_A = if a_is_colMajor: a.strides[1]
           else: a.strides[0]

    transpose_B = if b_is_colMajor: CUBLAS_OP_N
                  else: CUBLAS_OP_T
    ld_B = if b_is_colMajor: b.strides[1]
           else: b.strides[0]

  cublas_geam(transpose_A, transpose_B,
              a.shape[0], a.shape[1],
              alpha,
              a.get_data_ptr, ld_A,
              beta,
              b.get_data_ptr, ld_B,
              result.get_data_ptr, result.strides[1])

# ####################################################################
# BLAS Level 1 (Vector dot product, Addition, Scalar to Vector/Matrix)

proc `.*`*[T: SomeReal](a, b: CudaTensor[T]): T {.inline.}=
  ## Vector to Vector dot (scalar) product
  when compileOption("boundChecks"): check_dot_prod(a,b)
  cublas_dot( a.shape[0],
              a.get_data_ptr, a.strides[0],
              b.get_data_ptr, b.strides[0],
              addr result)

proc `+=`*[T: SomeReal](a: var CudaTensor[T], b: CudaTensor[T]) =
  ## Tensor in-place addition
  ## Only Vector-Vector and Matrix-Matrix addition are supported for now.
  ## For Matrix-Matrix, both matrices must have a contiguous layout.

  when compileOption("boundChecks"): check_add(a,b)

  if a.rank == 1:
    cudaVV_A_eq_A_p_bB(a, 1.T, b)
  elif a.rank == 2:
    cudaMM_A_eq_aA_p_bB(1.T, a, 1.T, b)
  else:
    raise newException(ValueError, "NotImplemented: Tensor addition is not implemented for 3D+ tensors")

  # TODO: if a and b share the same location, copy a to a new location
  # a += transpose(a) fails with CUBLAS ERROR 7.

proc `+`*[T: SomeReal](a,b: CudaTensor[T]): CudaTensor[T] =
  ## Tensor addition
  ## Only Vector-Vector and Matrix-Matrix addition are supported for now
  ## For Matrix-Matrix, both matrices must have a contiguous layout.

  when compileOption("boundChecks"): check_add(a,b)

  if a.rank == 1:
    cudaVV_C_eq_A_p_bB(a, 1.T, b, result)
  elif a.rank == 2:
    cudaMM_C_eq_aA_p_aB(1.T, a, 1.T, b, result)
  else:
    raise newException(ValueError, "NotImplemented: Tensor addition is not implemented for 3D+ tensors")


proc `-=`*[T: SomeReal](a: var CudaTensor[T], b: CudaTensor[T]) =
  ## Tensor in-place substraction
  ## Only Vector-Vector and Matrix-Matrix addition are supported for now.
  ## For Matrix-Matrix, both matrices must have a contiguous layout.

  when compileOption("boundChecks"): check_add(a,b)

  if a.rank == 1:
    cudaVV_A_eq_A_p_bB(a, -1.T, b)
  elif a.rank == 2:
    cudaMM_A_eq_aA_p_bB(1.T, a, -1.T, b)
  else:
    raise newException(ValueError, "NotImplemented: Tensor addition is not implemented for 3D+ tensors")

  # TODO: if a and b share the same location, copy a to a new location
  # a -= transpose(a) fails with CUBLAS ERROR 7.

proc `-`*[T: SomeReal](a,b: CudaTensor[T]): CudaTensor[T] =
  ## Tensor substraction
  ## Only Vector-Vector and Matrix-Matrix addition are supported for now
  ## For Matrix-Matrix, both matrices must have a contiguous layout.

  when compileOption("boundChecks"): check_add(a,b)

  if a.rank == 1:
    cudaVV_C_eq_A_p_bB(a, -1.T, b, result)
  elif a.rank == 2:
    cudaMM_C_eq_aA_p_aB(1.T, a, -1.T, b, result)
  else:
    raise newException(ValueError, "NotImplemented: Tensor addition is not implemented for 3D+ tensors")

proc `*=`*[T:SomeReal](t: var CudaTensor[T]; a: T) {.inline.}=
  ## Tensor inplace multiplication by a scalar

  # We multiply all elements of the CudaTensor regardless of shape/strides
  # So this operation can be applied to tensors of all ranks.
  # Hence we use the whole allocated length and a stride of 1
  cublas_scal(t.len, a, t.get_data_ptr, 1)

proc `*`*[T:SomeReal](a: T, t: CudaTensor[T]): CudaTensor[T] {.inline.}=
  ## Tensor multiplication by a scalar

  result = t.clone()
  result *= a

proc `*`*[T:SomeReal](t: CudaTensor[T], a: T): CudaTensor[T] {.inline.}=
  ## Tensor multiplication by a scalar
  a * t

proc `/=`*[T:SomeReal](t: var CudaTensor[T]; a: T) {.inline.}=
  ## Tensor in-place division by a scalar
  t *= (1/a)

proc `/`*[T:SomeReal](t: CudaTensor[T], a: T): CudaTensor[T] {.inline.}=
  ## Tensor division by a scalar
  (1/a) * t

proc `/`*[T:SomeReal](a: T, t: CudaTensor[T]): CudaTensor[T] {.inline.}=
  ## Tensor division by a scalar
  (1/a) * t