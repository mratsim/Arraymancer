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

when defined(blis):
  import ../backend/blis

import  # ../fallback/legacy/blas_l3_gemm, # Replaced by laser
        ../../laser/primitives/matrix_multiplication/gemm,
        ../data_structure,
        nimblas
from complex import Complex

# #################################################
# BLAS Level 2 (Matrix-Vector)

when defined(blis):
  proc blisMV_y_eq_aAx_p_by*[T: SomeFloat](
    alpha: T, a, x: Tensor[T],
    beta: T, y: var Tensor[T]) {.inline,noSideEffect.}=
    # Matrix-Vector: y = alpha A matvecmul x + beta y

    # Note: bounds checking must be done by the calling proc

    let
      M = a.shape[0]
      N = a.shape[1] # = x.shape[0], x is considered as a column vector

    bli_gemv(
        BLIS_NO_TRANSPOSE,
        BLIS_NO_CONJUGATE,
        M, N,
        unsafeAddr(alpha),
        a.get_offset_ptr, a.strides[0], a.strides[1],
        x.get_offset_ptr, x.strides[0],
        unsafeAddr(beta),
        y.get_offset_ptr, y.strides[0],
        )

# Note the fallback for non-real "naive_gemv_fallback" is called directly

proc blasMV_y_eq_aAx_p_by*[T: SomeFloat|Complex[float32]|Complex[float64]](
  alpha: T, a, x: Tensor[T],
  beta: T, y: var Tensor[T]) =
  # Matrix-Vector: y = alpha A matvecmul x + beta y

  # Note: bounds checking must be done by the calling proc
  # If needed, we trick BLAS to get a rowMajor result

  let
    M = a.shape[0]
    N = a.shape[1] # = x.shape[0], x is considered as a column vector

    cont_A = a.asContiguous # if not contiguous, change to row Major
    cont_A_is_rowMajor = cont_A.is_C_contiguous

    cont_A_order =  if cont_A_is_rowMajor: rowMajor
                    else: colMajor

    lda =  if cont_A_is_rowMajor: N # leading dimension
            else: M

  when type(alpha) is Complex:
    gemv( cont_A_order, noTranspose,
          M, N,
          unsafeAddr(alpha), cont_A.get_offset_ptr, lda,
          x.get_offset_ptr, x.strides[0],
          unsafeAddr(beta), y.get_offset_ptr, y.strides[0])
  else:
    gemv( cont_A_order, noTranspose,
          M, N,
          alpha, cont_A.get_offset_ptr, lda,
          x.get_offset_ptr, x.strides[0],
          beta, y.get_offset_ptr, y.strides[0])



# #################################################
# BLAS Level 3 (Matrix-Matrix)

when defined(blis):
  proc blisMM_C_eq_aAB_p_bC*[T: SomeFloat](
    alpha: T, a, b: Tensor[T],
    beta: T, c: var Tensor[T]) {.inline,noSideEffect.}=
    # Matrix: C = alpha A matmul B + beta C
    let
      M = a.shape[0]
      K = a.shape[1] # = b.shape[0]
      N = b.shape[1]

    # General stride-aware Matrix Multiply from BLIS.
    bli_gemm(BLIS_NO_TRANSPOSE, BLIS_NO_TRANSPOSE,
        M, N, K,
        unsafeAddr(alpha),
        a.get_offset_ptr, a.strides[0], a.strides[1],
        b.get_offset_ptr, b.strides[0], b.strides[1],
        unsafeAddr(beta),
        c.get_offset_ptr, c.strides[0], c.strides[1])

proc fallbackMM_C_eq_aAB_p_bC*[T: SomeInteger](
  alpha: T, a, b: Tensor[T],
  beta: T, c: var Tensor[T]) {.inline.}=
  # Matrix: C = alpha A matmul B + beta C
  let
    M = a.shape[0]
    K = a.shape[1] # = b.shape[0]
    N = b.shape[1]

  # Legacy fallback
  # gemm_nn_fallback( M, N, K,
  #                   alpha,
  #                   a.data, a.offset,
  #                   a.strides[0], a.strides[1],
  #                   b.data, b.offset,
  #                   b.strides[0], b.strides[1],
  #                   beta,
  #                   c.data, c.offset,
  #                   c.strides[0], c.strides[1])

  # Laser backend
  gemm_strided(M, N, K,
               alpha,
               a.get_offset_ptr(),
               a.strides[0], a.strides[1],
               b.get_offset_ptr(),
               b.strides[0], b.strides[1],
               beta,
               c.get_offset_ptr(),
               c.strides[0], c.strides[1])

proc blasMM_C_eq_aAB_p_bC*[T: SomeFloat|Complex[float32]|Complex[float64]](
  alpha: T, a, b: Tensor[T],
  beta: T, c: var Tensor[T]) =
  # Matrix: C = alpha A matmul B + beta C

  let
    M = a.shape[0]
    K = a.shape[1] # b.shape[0]
    N = b.shape[1]

    A = a.asContiguous()
    B = b.asContiguous()

  assert c.isContiguous()

  var lda, ldb: int
  var tA, tb: TransposeType

  # General Matrix Multiply from nimblas.
  if c.is_C_contiguous():   # [M, N]
    if A.is_C_contiguous(): # [M, K]
      lda = K
      tA = noTranspose
    else:
      lda = M
      tA = transpose
    if B.is_C_contiguous(): # [K, N]
      ldb = N
      tB = noTranspose
    else:
      ldb = K
      tB = transpose

    when type(alpha) is Complex:
      gemm(rowMajor,
           tA, tB,
           M, N, K,
           unsafeAddr(alpha), A.get_offset_ptr, lda,
                              B.get_offset_ptr, ldb,
           unsafeAddr(beta),  c.get_offset_ptr, N)
    else:
      gemm(rowMajor,
           tA, tB,
           M, N, K,
           alpha, A.get_offset_ptr, lda,
                  B.get_offset_ptr, ldb,
           beta,  c.get_offset_ptr, N)
  else: # column major result [M, N] - TODO tests
    if A.is_C_contiguous(): # [M, K]
      lda = K
      tA = transpose
    else:
      lda = M
      tA = noTranspose
    if B.is_C_contiguous(): # [K, N]
      ldb = N
      tB = transpose
    else:
      ldb = K
      tB = noTranspose

    when typeof(alpha) is Complex:
      gemm(colMajor,
           tA, tB,
           M, N, K,
           unsafeAddr(alpha), A.get_offset_ptr, lda,
                              B.get_offset_ptr, ldb,
           unsafeAddr(beta),  c.get_offset_ptr, M)
    else:
      gemm(colMajor,
           tA, tB,
           M, N, K,
           alpha, A.get_offset_ptr, lda,
                  B.get_offset_ptr, ldb,
           beta,  c.get_offset_ptr, M)
