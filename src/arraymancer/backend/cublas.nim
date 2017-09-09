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

# L1 BLAS

# Vector copy
proc cublas_copy[T: SomeReal](
  n: int; x: ptr T; incx: int;
  y: ptr T; incy: int): cublasStatus_t {.inline.}=

  check cublasSetStream(defaultHandle, defaultStream)

  when T is float32:
    cublasScopy(defaultHandle, n.cint, x, incx.cint, y, incy.cint)
  elif T is float64:
    cublasDcopy(defaultHandle, n.cint, x, incx.cint, y, incy.cint)
  else:
    raise newException(ValueError, "Unreachable")

# Vector dot product
proc cublas_dot[T: SomeReal](
  n: int;
  x: ptr T; incx: int;
  y: ptr T; incy: int;
  output: ptr T): cublasStatus_t {.inline.}=

  check cublasSetStream(defaultHandle, defaultStream)

  when T is float32:
    cublasSdot(defaultHandle, n.cint, x, incx.cint, y, incy.cint, output)
  elif T is float64:
    cublasDdot(defaultHandle, n.cint, x, incx.cint, y, incy.cint, output)
  else:
    raise newException(ValueError, "Unreachable")

# Vector addition
proc cublas_axpy[T: SomeReal](
  n: int;
  alpha: ptr T;
  x: ptr T; incx: int;
  y: ptr T; incy: int): cublasStatus_t {.inline.}=

  check cublasSetStream(defaultHandle, defaultStream)

  when T is float32:
    cublasSaxpy(defaultHandle, n.cint, alpha, x, incx.cint, y, incy.cint)
  elif T is float64:
    cublasDaxpy(defaultHandle, n.cint, alpha, x, incx.cint, y, incy.cint)
  else:
    raise newException(ValueError, "Unreachable")

# Scalar multiplication
proc cublas_scal[T: SomeReal](
  n: int; alpha: ptr T;
  x: ptr T; incx: int): cublasStatus_t {.inline.}=

  check cublasSetStream(defaultHandle, defaultStream)

  when T is float32:
    cublasSscal(defaultHandle, n.cint, alpha, x, incx.cint)
  elif T is float64:
    cublasDscal(defaultHandle, n.cint, alpha, x, incx.cint)
  else:
    raise newException(ValueError, "Unreachable")

# BLAS extension (L1-like)
# Matrix addition (non-standard BLAS)
proc cublas_geam[T: SomeReal](
  transa, transb: cublasOperation_t;
  m, n: int;
  alpha: ptr T; A: ptr T; lda: int;
  beta: ptr T; B: ptr T; ldb: int;
  C: ptr T; ldc: int): cublasStatus_t {.inline.}=

  check cublasSetStream(defaultHandle, defaultStream)

  when T is float32:
    cublasSgeam(defaultHandle,
            transa, transb, m.cint, n.cint,
            alpha, A, lda.cint,
            beta, B, ldb.cint,
            C, ldc.cint)
  elif T is float64:
    cublasDgeam(defaultHandle,
                transa, transb, m.cint, n.cint,
                alpha, A, lda.cint,
                beta, B, ldb.cint,
                C, ldc.cint)
  else:
    raise newException(ValueError, "Unreachable")

# L2 BLAS

# L3 BLAS
proc cublas_gemm*[T: SomeReal](
  transa, transb: cublasOperation_t,
  m, n, k: int,
  alpha: ptr T, A: ptr T, lda: int,
  B: ptr T; ldb: int;
  beta: ptr T; C: ptr T; ldc: int): cublasStatus_t {.inline.}=

  check cublasSetStream(defaultHandle, defaultStream)

  when T is float32:
    cublasSgemm(defaultHandle,
                transa, transb,
                m.cint, n.cint, k.cint,
                alpha, A, lda.cint,
                B, ldb.cint,
                beta, C, ldc.cint)
  elif T is float64:
    cublasDgemm(defaultHandle,
                transa, transb,
                m.cint, n.cint, k.cint,
                alpha, A, lda.cint,
                B, ldb.cint,
                beta, C, ldc.cint)
