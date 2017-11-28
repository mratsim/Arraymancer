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

import  nimcuda/[cublas_v2, cublas_api],
        ./cuda_global_state,
        ./cuda


export cuda, cublas_v2, cublas_api

# #####################################################
# Redefinition of imported CUDA proc with standard name
# and use of streams for parallel async processing

# L1 BLAS

# Vector copy
proc cublas_copy*[T: SomeReal](
  n: int; x: ptr T; incx: int;
  y: ptr T; incy: int) {.inline.}=

  check cublasSetStream(cublasHandle0, cudaStream0)

  when T is float32:
    check cublasScopy(cublasHandle0, n.cint, x, incx.cint, y, incy.cint)
  elif T is float64:
    check cublasDcopy(cublasHandle0, n.cint, x, incx.cint, y, incy.cint)

# Vector dot product
proc cublas_dot*[T: SomeReal](
  n: int;
  x: ptr T; incx: int;
  y: ptr T; incy: int;
  output: ptr T) {.inline.}=

  check cublasSetStream(cublasHandle0, cudaStream0)

  when T is float32:
    check cublasSdot(cublasHandle0, n.cint, x, incx.cint, y, incy.cint, output)
  elif T is float64:
    check cublasDdot(cublasHandle0, n.cint, x, incx.cint, y, incy.cint, output)

# Vector addition
proc cublas_axpy*[T: SomeReal](
  n: int;
  alpha: T;
  x: ptr T; incx: int;
  y: ptr T; incy: int) {.inline.}=
  # Y = alpha X + Y
  # X, Y: vectors

  var al = alpha

  check cublasSetStream(cublasHandle0, cudaStream0)

  when T is float32:
    check cublasSaxpy(cublasHandle0, n.cint, addr al, x, incx.cint, y, incy.cint)
  elif T is float64:
    check cublasDaxpy(cublasHandle0, n.cint, addr al, x, incx.cint, y, incy.cint)

# Scalar multiplication
proc cublas_scal*[T: SomeReal](
  n: int; alpha: T;
  x: ptr T; incx: int) {.inline.}=

  var al = alpha

  check cublasSetStream(cublasHandle0, cudaStream0)

  when T is float32:
    check cublasSscal(cublasHandle0, n.cint, addr al, x, incx.cint)
  elif T is float64:
    check cublasDscal(cublasHandle0, n.cint, addr al, x, incx.cint)

# BLAS extension (L1-like)
# Matrix addition (non-standard BLAS)
proc cublas_geam*[T: SomeReal](
  transa, transb: cublasOperation_t;
  m, n: int;
  alpha: T; A: ptr T; lda: int;
  beta: T; B: ptr T; ldb: int;
  C: ptr T; ldc: int) {.inline.}=

  var
    al = alpha
    be = beta

  check cublasSetStream(cublasHandle0, cudaStream0)

  when T is float32:
    check cublasSgeam(cublasHandle0,
                      transa, transb, m.cint, n.cint,
                      addr al, A, lda.cint,
                      addr be, B, ldb.cint,
                      C, ldc.cint)
  elif T is float64:
    check cublasDgeam(cublasHandle0,
                      transa, transb, m.cint, n.cint,
                      addr al, A, lda.cint,
                      addr be, B, ldb.cint,
                      C, ldc.cint)

# L2 BLAS
proc cublas_gemv*[T: SomeReal](
  trans: cublasOperation_t, m, n: int,
  alpha: T, A: ptr T, lda: int,
  x: ptr T, incx: int,
  beta: T, y: ptr T, incy: int) {.inline.}=
  # y = alpha A * x + beta y
  # A: matrix
  # x, y: vectors

  var
    al = alpha
    be = beta

  check cublasSetStream(cublasHandle0, cudaStream0)

  when T is float32:
    check cublasSgemv(cublasHandle0,
                      trans, m.cint, n.cint,
                      addr al, A, lda.cint, x, incx.cint,
                      addr be, y, incy.cint)
  elif T is float64:
    check cublasDgemv(cublasHandle0,
                      trans, m.cint, n.cint,
                      addr al, A, lda.cint, x, incx.cint,
                      addr be, y, incy.cint)

# L3 BLAS
proc cublas_gemm*[T: SomeReal](
  transa, transb: cublasOperation_t,
  m, n, k: int,
  alpha: T, A: ptr T, lda: int,
  B: ptr T; ldb: int;
  beta: T; C: ptr T; ldc: int) {.inline.}=
  # C = alpha A * B + beta C
  # A, B, C: matrices

  var
    al = alpha
    be = beta

  check cublasSetStream(cublasHandle0, cudaStream0)

  when T is float32:
    check cublasSgemm(cublasHandle0,
                      transa, transb,
                      m.cint, n.cint, k.cint,
                      addr al, A, lda.cint,
                      B, ldb.cint,
                      addr be, C, ldc.cint)
  elif T is float64:
    check cublasDgemm(cublasHandle0,
                      transa, transb,
                      m.cint, n.cint, k.cint,
                      addr al, A, lda.cint,
                      B, ldb.cint,
                      addr be, C, ldc.cint)

proc cublas_gemmStridedBatched*[T: SomeReal](
  transa, transb: cublasOperation_t;
  m, n, k: int;
  alpha: T; A: ptr T; lda: int; strideA: int;
  B: ptr T; ldb: int; strideB: int;
  beta: T; C: ptr T; ldc: int; strideC: int;
  batchCount: int) {.inline.} =
  # C + i*strideC = αop(A + i*strideA)op(B + i*strideB)+β(C + i*strideC),
  # for i  ∈ [ 0 , b a t c h C o u n t − 1 ]
  # A, B, C: matrices

  var
    al = alpha
    be = beta

  check cublasSetStream(cublasHandle0, cudaStream0)

  when T is float32:
    check cublasSgemmStridedBatched(
      cublasHandle0,
      transa, transb,
      m.cint, n.cint, k.cint,
      addr al, A, lda.cint, strideA,
      B, ldb.cint, strideB,
      addr be, C, ldc.cint, strideC,
      batchCount.cint
    )
  elif T is float64:
    check cublasDgemmStridedBatched(
      cublasHandle0,
      transa, transb,
      m.cint, n.cint, k.cint,
      addr al, A, lda.cint, strideA,
      B, ldb.cint, strideB,
      addr be, C, ldc.cint, strideC,
      batchCount.cint
    )
