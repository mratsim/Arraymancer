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

proc cudaMM_C_eq_aAB_mul_bC[T: SomeReal](
  alpha: T, a, b: CudaTensor[T],
  beta: T, c: var CudaTensor[T]) =
  # Matrix: C = alpha A matmul B + beta C

  # TODO: remove this contiguous layout constraint (via conversion or custom kernel)
  if not (a.isContiguous and b.isContiguous):
    raise newException(ValueError, "NotImplemented: for now both tensors should be contiguous")

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
    
    ld_C = c.strides[1] # C is always F contiguous (TODO test)

  cublas_gemm(transpose_A, transpose_B,
              a.shape[0], b.shape[1], a.shape[1],
              alpha, a.get_data_ptr, ld_A,
              b.get_data_ptr, ld_B,
              beta, c.get_data_ptr, ld_C)

proc `*`*[T: SomeReal](a, b: CudaTensor[T]): CudaTensor[T] =
  ## Matrix multiplication (Matrix-Matrix and Matrix-Vector) on CUDA

  if a.rank == 2 and b.rank == 2:
    when compileOption("boundChecks"): check_matmat(a,b)
    result = newCudaTensor[T]([a.shape[0], b.shape[1]])
    cudaMM_C_eq_aAB_mul_bC(1.T, a, b, 0.T, result)
  # elif a.rank == 2 and b.rank == 1:  matvec_blas(a, b, result)
  else: raise newException(ValueError, "Matrix-Matrix or Matrix-Vector multiplication valid only if first Tensor is a Matrix and second is a Matrix or Vector")