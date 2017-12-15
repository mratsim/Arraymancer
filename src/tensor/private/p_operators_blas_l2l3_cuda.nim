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


# Cublas helper procs for L1 BLAS
# With custom kernels they shouldn't be needed anymore
# They however have a nice interface to call for fused aX + Y or aA + Bb

#################################################
## In-place

proc cudaVV_A_eq_A_p_bB[T: SomeReal](
  a: var CudaTensor[T], beta: T, b: CudaTensor[T]) {.inline, deprecated.}=
  # Vector: A = A + beta B

  cublas_axpy(a.shape[0],
              beta,
              b.get_data_ptr, b.strides[0],
              a.get_data_ptr, a.strides[0])

proc cudaMM_A_eq_aA_p_bB[T: SomeReal](
  alpha: T, a: var CudaTensor[T],
  beta: T, b: CudaTensor[T]) {.deprecated.}=
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

#############################################################
## Out-of-place

proc cudaVV_C_eq_A_p_bB[T: SomeReal]( a: CudaTensor[T],
                                      beta: T, b: CudaTensor[T],
                                      result: var CudaTensor[T]) {.inline, deprecated.}=
  # Vector: C = A + beta B
  result = newCudaTensor[T](a.shape)

  cublas_copy(a.len, a.get_data_ptr, a.strides[0],
              result.get_data_ptr, result.strides[0])

  cudaVV_A_eq_A_p_bB(result, beta, b)

proc cudaMM_C_eq_aA_p_aB[T: SomeReal](alpha: T, a: CudaTensor[T],
                                          beta: T, b: CudaTensor[T],
                                          result: var CudaTensor[T]) {.deprecated.}=
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