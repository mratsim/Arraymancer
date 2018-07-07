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

import  ./backend/cublas,
        ./private/p_init_cuda,
        ./private/p_checks,
        ./data_structure

proc cudaMV_y_eq_aAx_p_by[T: SomeFloat](
  alpha: T, a, x: CudaTensor[T],
  beta: T, y: var CudaTensor[T]) =
  # Matrix-Vector: y = alpha A matvecmul x + beta y

  # TODO: remove this contiguous layout constraint
  if not a.isContiguous:
    raise newException(ValueError, "NotImplemented: for now both tensors should be contiguous")

  let
    a_is_colMajor = a.is_F_contiguous

    transpose_A = if a_is_colMajor: CUBLAS_OP_N
                  else: CUBLAS_OP_T
    lda = if a_is_colMajor: a.strides[1]
          else: a.strides[0]

  cublas_gemv(
      transpose_A, a.shape[0], a.shape[1],
      alpha, a.get_data_ptr, lda,
      x.get_data_ptr, x.strides[0],
      beta, y.get_data_ptr, y.strides[0])

proc cudaMM_C_eq_aAB_p_bC[T: SomeFloat](
  alpha: T, a, b: CudaTensor[T],
  beta: T, c: var CudaTensor[T]) =
  # Matrix: C = alpha A matmul B + beta C

  # TODO: remove this contiguous layout constraint
  if not (a.isContiguous and b.isContiguous):
    raise newException(ValueError, "NotImplemented: for now both tensors should be contiguous")

  let
    a_is_colMajor = a.is_F_contiguous
    b_is_colMajor = b.is_F_contiguous

    transpose_A = if a_is_colMajor: CUBLAS_OP_N
                  else: CUBLAS_OP_T
    lda = if a_is_colMajor: a.strides[1]
          else: a.strides[0]

    transpose_B = if b_is_colMajor: CUBLAS_OP_N
                  else: CUBLAS_OP_T
    ldb = if b_is_colMajor: b.strides[1]
          else: b.strides[0]

    ldc = c.strides[1] # C is always F contiguous (TODO test)

  cublas_gemm(transpose_A, transpose_B,
              a.shape[0], b.shape[1], a.shape[1],
              alpha, a.get_data_ptr, lda,
              b.get_data_ptr, ldb,
              beta, c.get_data_ptr, ldc)

proc `*`*[T: SomeFloat](a, b: CudaTensor[T]): CudaTensor[T] =
  ## Matrix multiplication (Matrix-Matrix and Matrix-Vector) on CUDA

  if a.rank == 2 and b.rank == 2:
    when compileOption("boundChecks"):
      check_matmat(a,b)
    result = newCudaTensor[T]([a.shape[0], b.shape[1]])
    cudaMM_C_eq_aAB_p_bC(1.T, a, b, 0.T, result)
  elif a.rank == 2 and b.rank == 1:
    when compileOption("boundChecks"):
      check_matvec(a,b)
    result = newCudaTensor[T]([a.shape[0]])
    cudaMV_y_eq_aAx_p_by(1.T,a, b, 0.T, result)
  else: raise newException(ValueError, "Matrix-Matrix or Matrix-Vector multiplication valid only if first Tensor is a Matrix and second is a Matrix or Vector")
