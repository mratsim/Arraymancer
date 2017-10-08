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

when defined(blis):
  import ./backend/blis

import  ../../private/sequninit,
        ./p_checks,
        ../fallback/blas_l3_gemm,
        ../fallback/naive_l2_gemv,
        ../data_structure,
        ../data_structure_helpers,
        nimblas

# #################################################
# BLAS Level 2 (Matrix-Vector)

template matvec_blis*[T: SomeReal](a, x, result: Tensor[T]): auto =
  ## Matrix to Vector Multiply for float tensors of rank 2 and 1
  let
    rowA = a.shape[0]
    colA = a.shape[1]
    rowX = x.shape[0] # X is considered as a column vector

  when compileOption("boundChecks"):
    check_matvec(a,b)

  result.data = newSeqUninit[T](rowA)
  result.shape = [rowA].toMetadataArray
  result.strides = [1].toMetadataArray
  result.offset = 0

  let
    a_ptr = get_data_ptr(a)
    x_ptr = get_data_ptr(x)
    res_ptr = get_data_ptr(result)
    alpha = 1.T
    alpha_ptr = unsafeAddr(alpha)
    beta = 0.T
    beta_ptr = unsafeAddr(beta)

  # General stride-aware Matrix-Vector Multiply from BLIS.
  bli_gemv(
      BLIS_NO_TRANSPOSE,
      BLIS_NO_CONJUGATE,
      rowA, rowX,
      alpha_ptr,
      a_ptr, a.strides[0], a.strides[1],
      x_ptr, x.strides[0],
      beta_ptr,
      res_ptr, 1,
      )

template matvec_blas*[T: SomeReal](a, x, result: Tensor[T]): auto =
  ## Matrix to Vector Multiply for float tensors of rank 2 and 1
  let
    rowA = a.shape[0]
    colA = a.shape[1]
    rowX = x.shape[0] # X is considered as a column vector

  when compileOption("boundChecks"): check_matvec(a,b)

  result.data = newSeqUninit[T](rowA)
  result.shape = [rowA].toMetadataArray
  result.strides = [1].toMetadataArray
  result.offset = 0

  # Stride for X is supported via incx argument of GEMV
  let cont_a = a.unsafeContiguous

  let a_ptr = get_data_ptr(cont_a)
  let x_ptr = get_data_ptr(x)
  let res_ptr = get_data_ptr(result)

  let a_tr = getTransposeTarget(cont_a)

  # General Matrix-Vector Multiply from nimblas.
  if a_tr == TransposeType.noTranspose: # A is rowMajor
    gemv(rowMajor, a_tr, rowA, rowX, 1, a_ptr, colA, x_ptr, x.strides[0], 0, res_ptr, 1)
  else: # A is colMajor
    gemv(colMajor, noTranspose, rowA, rowX, 1, a_ptr, rowA, x_ptr, x.strides[0], 0, res_ptr, 1)

template matvec_fallback*[T: SomeInteger](a, x, result: Tensor[T]): auto =
  let rowA = a.shape[0]

  when compileOption("boundChecks"): check_matvec(a,b)

  result.data = newSeqUninit[T](rowA)
  result.shape = [rowA].toMetadataArray
  result.strides = [1].toMetadataArray
  result.offset = 0

  naive_gemv_fallback(
    1.T,
    a,
    x,
    0.T,
    result
  )

# #################################################
# BLAS Level 3 (Matrix-Matrix)

template matmat_blis*[T: SomeReal](a, b, result: Tensor[T]): auto =
  ## Matrix to matrix Multiply for float tensors of rank 2
  let
    rowA = a.shape[0]
    colA = a.shape[1]
    rowB = b.shape[0]
    colB = b.shape[1]

  when compileOption("boundChecks"): check_matmat(a,b)

  result.data = newSeqUninit[T](rowA * colB)
  result.shape = [rowA, colB].toMetadataArray
  result.strides = [rowA, 1].toMetadataArray  # We force row-major after computation
  result.offset = 0

  let
    a_ptr = get_data_ptr(a)
    b_ptr = get_data_ptr(b)
    res_ptr = get_data_ptr(result)
    alpha = 1.T
    alpha_ptr = unsafeAddr(alpha)
    beta = 0.T
    beta_ptr = unsafeAddr(beta)

  # General stride-aware Matrix Multiply from BLIS.
  bli_gemm(BLIS_NO_TRANSPOSE, BLIS_NO_TRANSPOSE,
      rowA, colB, rowB,
      alpha_ptr,
      a_ptr, a.strides[0], a.strides[1],
      b_ptr, b.strides[0], b.strides[1],
      beta_ptr,
      res_ptr, result.strides[0], 1)

template matmat_blas*[T: SomeReal](a, b, result: Tensor[T]): auto =
  ## Matrix to matrix Multiply for float tensors of rank 2
  let
    M = a.shape[0]
    K = a.shape[1] # b.shape[0]
    N = b.shape[1]

  when compileOption("boundChecks"): check_matmat(a,b)

  result.data = newSeqUninit[T](M * N)
  result.shape = [M, N].toMetadataArray
  result.strides = [N, 1].toMetadataArray
  result.offset = 0

  # TODO use a GEMM kernel that supports strided arrays like BLIS
  # That avoids copies and a conversion step
  let
    cont_a = a.unsafeContiguous
    cont_b = b.unsafeContiguous

    a_ptr = get_data_ptr(cont_a)
    b_ptr = get_data_ptr(cont_b)
    res_ptr = get_data_ptr(result)

    a_tr = getTransposeTarget(cont_a)
    b_tr = getTransposeTarget(cont_b)

  # General Matrix Multiply from nimblas.
  if a_tr == TransposeType.noTranspose and b_tr == TransposeType.noTranspose:
    gemm(rowMajor, a_tr, b_tr, M, N, K, 1, a_ptr, K, b_ptr, N, 0, res_ptr, N)
  elif a_tr == TransposeType.transpose and b_tr == TransposeType.noTranspose:
    gemm(rowMajor, a_tr, b_tr, M, N, K, 1, a_ptr, M, b_ptr, N, 0, res_ptr, N)
  elif a_tr == TransposeType.noTranspose and b_tr == TransposeType.transpose:
    gemm(rowMajor, a_tr, b_tr, M, N, K, 1, a_ptr, K, b_ptr, K, 0, res_ptr, N)
  elif a_tr == TransposeType.transpose and b_tr == TransposeType.transpose:
    gemm(rowMajor, a_tr, b_tr, M, N, K, 1, a_ptr, M, b_ptr, K, 0, res_ptr, N)
  else: raise newException(ValueError, "The transposes types: " & $a_tr & " or " & $b_tr & " is not supported")

template matmat_fallback*[T: SomeInteger](a, b, result: Tensor[T]): auto =
  let M = a.shape[0]
  let K = a.shape[1]
  let N = b.shape[1]

  when compileOption("boundChecks"): check_matmat(a,b)

  result.shape = [M, N].toMetadataArray
  result.strides = [N, 1].toMetadataArray
  result.offset = 0
  result.data = newSeqUninit[T](M*N)

  gemm_nn_fallback(M, N, K,
           1.T,
           a.data, a.offset,
           a.strides[0], a.strides[1],
           b.data, b.offset,
           b.strides[0], b.strides[1],
           0.T,
           result.data, 0,
           N, 1)