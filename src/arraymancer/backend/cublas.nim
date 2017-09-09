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

# #####################################################
# Redefinition of imported CUDA proc with standard name
# and use of streams for parallel async processing

# Vector copy
proc cublas_copy(n: int; x: ptr float32; incx: int;
                         y: ptr float32; incy: int): cublasStatus_t =

  check cublasSetStream(defaultHandle, defaultStream)
  cublasScopy(defaultHandle, n.cint, x, incx.cint, y, incy.cint)

proc cublas_copy(n: int; x: ptr float64; incx: int;
                         y: ptr float64; incy: int): cublasStatus_t =

  check cublasSetStream(defaultHandle, defaultStream)
  cublasDcopy(defaultHandle, n.cint, x, incx.cint, y, incy.cint)

# Vector dot product
proc cublas_dot(n: int;
                x: ptr float32; incx: int;
                y: ptr float32; incy: int;
                output: ptr float32): cublasStatus_t =

  check cublasSetStream(defaultHandle, defaultStream)
  cublasSdot(defaultHandle, n.cint, x, incx.cint, y, incy.cint, output)

proc cublas_dot(n: int;
                x: ptr float64; incx: int;
                y: ptr float64; incy: int;
                output: ptr float64): cublasStatus_t =

  check cublasSetStream(defaultHandle, defaultStream)
  cublasDdot(defaultHandle, n.cint, x, incx.cint, y, incy.cint, output)

# Vector addition
proc cublas_axpy( n: int;
                  alpha: ptr float32;
                  x: ptr float32; incx: int;
                  y: ptr float32; incy: int): cublasStatus_t =

  check cublasSetStream(defaultHandle, defaultStream)
  cublasSaxpy(defaultHandle, n.cint, alpha, x, incx.cint, y, incy.cint)

proc cublas_axpy( n: int;
                  alpha: ptr float64;
                  x: ptr float64; incx: int;
                  y: ptr float64; incy: int): cublasStatus_t =

  check cublasSetStream(defaultHandle, defaultStream)
  cublasDaxpy(defaultHandle, n.cint, alpha, x, incx.cint, y, incy.cint)

# Matrix addition (non-standard BLAS)
proc cublas_geam( transa, transb: cublasOperation_t;
                  m, n: int;
                  alpha: ptr float32; A: ptr float32; lda: int;
                  beta: ptr float32; B: ptr float32; ldb: int;
                  C: ptr float32; ldc: int): cublasStatus_t =

  check cublasSetStream(defaultHandle, defaultStream)
  cublasSgeam(defaultHandle,
              transa, transb, m.cint, n.cint,
              alpha, A, lda.cint,
              beta, B, ldb.cint,
              C, ldc.cint)

proc cublas_geam( transa, transb: cublasOperation_t;
                  m, n: int;
                  alpha: ptr float64; A: ptr float64; lda: int;
                  beta: ptr float64; B: ptr float64; ldb: int;
                  C: ptr float64; ldc: int): cublasStatus_t =

  check cublasSetStream(defaultHandle, defaultStream)
  cublasDgeam(defaultHandle,
              transa, transb, m.cint, n.cint,
              alpha, A, lda.cint,
              beta, B, ldb.cint,
              C, ldc.cint)

# Scalar multiplication
proc cublas_scal( n: int; alpha: ptr float32;
                  x: ptr float32; incx: int): cublasStatus_t =

  check cublasSetStream(defaultHandle, defaultStream)
  cublasSscal(defaultHandle, n.cint, alpha, x, incx.cint)

proc cublas_scal( n: int; alpha: ptr float64;
                  x: ptr float64; incx: int): cublasStatus_t =

  check cublasSetStream(defaultHandle, defaultStream)
  cublasDscal(defaultHandle, n.cint, alpha, x, incx.cint)