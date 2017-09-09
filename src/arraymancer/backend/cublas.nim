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
proc copy(n: cint; x: ptr cfloat; incx: cint;
                   y: ptr cfloat; incy: cint): cublasStatus_t =

  check cublasSetStream(defaultHandle, defaultStream)
  cublasScopy(defaultHandle, n, x, incx, y, incy)

proc copy(n: cint; x: ptr cdouble; incx: cint;
                   y: ptr cdouble; incy: cint): cublasStatus_t =

  check cublasSetStream(defaultHandle, defaultStream)
  cublasDcopy(defaultHandle, n, x, incx, y, incy)

# Vector dot product
proc dot(n: cint;
         x: ptr cfloat; incx: cint;
         y: ptr cfloat; incy: cint;
         output: ptr cfloat): cublasStatus_t =

  check cublasSetStream(defaultHandle, defaultStream)
  cublasSdot(defaultHandle, n, x, incx, y, incy, output)

proc dot(n: cint;
         x: ptr cdouble; incx: cint;
         y: ptr cdouble; incy: cint;
         output: ptr cdouble): cublasStatus_t =

  check cublasSetStream(defaultHandle, defaultStream)
  cublasDdot(defaultHandle, n, x, incx, y, incy, output)

# Vector addition
proc axpy(n: cint;
          alpha: ptr cfloat;
          x: ptr cfloat; incx: cint;
          y: ptr cfloat; incy: cint): cublasStatus_t =

  check cublasSetStream(defaultHandle, defaultStream)
  cublasSaxpy(defaultHandle, n, alpha, x, incx, y, incy)

proc axpy(n: cint;
          alpha: ptr cdouble;
          x: ptr cdouble; incx: cint;
          y: ptr cdouble; incy: cint): cublasStatus_t =

  check cublasSetStream(defaultHandle, defaultStream)
  cublasDaxpy(defaultHandle, n, alpha, x, incx, y, incy)

# Matrix addition (non-standard BLAS)
proc geam(transa, transb: cublasOperation_t;
          m, n: cint;
          alpha: ptr cfloat; A: ptr cfloat; lda: cint;
          beta: ptr cfloat; B: ptr cfloat; ldb: cint;
          C: ptr cfloat; ldc: cint): cublasStatus_t =

  check cublasSetStream(defaultHandle, defaultStream)
  cublasSgeam(defaultHandle,
              transa, transb, m, n,
              alpha, A, lda,
              beta, B, ldb,
              C, ldc)

proc geam(transa, transb: cublasOperation_t;
          m, n: cint;
          alpha: ptr cdouble; A: ptr cdouble; lda: cint;
          beta: ptr cdouble; B: ptr cdouble; ldb: cint;
          C: ptr cdouble; ldc: cint): cublasStatus_t =

  check cublasSetStream(defaultHandle, defaultStream)
  cublasDgeam(defaultHandle,
              transa, transb, m, n,
              alpha, A, lda,
              beta, B, ldb,
              C, ldc)

# Scalar multiplication
proc scal(n: cint; alpha: ptr cfloat;
          x: ptr cfloat; incx: cint): cublasStatus_t =

  check cublasSetStream(defaultHandle, defaultStream)
  cublasSscal(defaultHandle, n, alpha, x, incx)

proc scal(n: cint; alpha: ptr cdouble;
          x: ptr cdouble; incx: cint): cublasStatus_t =

  check cublasSetStream(defaultHandle, defaultStream)
  cublasDscal(defaultHandle, n, alpha, x, incx)