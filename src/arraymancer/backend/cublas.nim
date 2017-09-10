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
  alpha: T;
  x: ptr T; incx: int;
  y: ptr T; incy: int): cublasStatus_t {.inline.}=
  # Y = alpha X + Y
  # X, Y: vectors

  # We need to pass an address to CuBLAS for beta
  # If the input is not a variable but a float directly
  # It won't have an address and can't be used by CUBLAS
  let al = alpha

  check cublasSetStream(defaultHandle, defaultStream)

  when T is float32:
    cublasSaxpy(defaultHandle, n.cint, al.unsafeAddr, x, incx.cint, y, incy.cint)
  elif T is float64:
    cublasDaxpy(defaultHandle, n.cint, al.unsafeAddr, x, incx.cint, y, incy.cint)
  else:
    raise newException(ValueError, "Unreachable")

# Scalar multiplication
proc cublas_scal[T: SomeReal](
  n: int; alpha: T;
  x: ptr T; incx: int): cublasStatus_t {.inline.}=

  # We need to pass an address to CuBLAS for beta
  # If the input is not a variable but a float directly
  # It won't have an address and can't be used by CUBLAS
  let al = alpha

  check cublasSetStream(defaultHandle, defaultStream)

  when T is float32:
    cublasSscal(defaultHandle, n.cint, al.unsafeAddr, x, incx.cint)
  elif T is float64:
    cublasDscal(defaultHandle, n.cint, al.unsafeAddr, x, incx.cint)
  else:
    raise newException(ValueError, "Unreachable")

# BLAS extension (L1-like)
# Matrix addition (non-standard BLAS)
proc cublas_geam[T: SomeReal](
  transa, transb: cublasOperation_t;
  m, n: int;
  alpha: T; A: ptr T; lda: int;
  beta: T; B: ptr T; ldb: int;
  C: ptr T; ldc: int): cublasStatus_t {.inline.}=

  let
    al = alpha
    be = beta

  check cublasSetStream(defaultHandle, defaultStream)

  when T is float32:
    cublasSgeam(defaultHandle,
            transa, transb, m.cint, n.cint,
            al.unsafeAddr, A, lda.cint,
            be.unsafeAddr, B, ldb.cint,
            C, ldc.cint)
  elif T is float64:
    cublasDgeam(defaultHandle,
                transa, transb, m.cint, n.cint,
            al.unsafeAddr, A, lda.cint,
            be.unsafeAddr, B, ldb.cint,
                C, ldc.cint)
  else:
    raise newException(ValueError, "Unreachable")

# L2 BLAS
proc cublasSgemv[T: SomeReal](
  trans: cublasOperation_t, m, n: int,
  alpha: T, A: ptr T, lda: int,
  x: ptr T, incx: int,
  beta: T, y: ptr T, incy: int): cublasStatus_t {.inline.}=
  # y = alpha A * x + beta y
  # A: matrix
  # x, y: vectors

  let
    al = alpha
    be = beta

  check cublasSetStream(defaultHandle, defaultStream)

  when T is float32:
    ublasSgemv*(defaultHandle,
                trans, m.cint, n.cint,
                al.unsafeAddr, A, lda.cint, x, incx.cint,
                be.unsafeAddr, y, incy.cint)
  elif T is float64:
    cublasDgemv*(defaultHandle,
                trans, m.cint, n.cint,
                al.unsafeAddr, A, lda.cint, x, incx.cint,
                be.unsafeAddr, y, incy.cint)

# L3 BLAS
proc cublas_gemm*[T: SomeReal](
  transa, transb: cublasOperation_t,
  m, n, k: int,
  alpha: T, A: ptr T, lda: int,
  B: ptr T; ldb: int;
  beta: T; C: ptr T; ldc: int): cublasStatus_t {.inline.}=
  # C = alpha A * B + beta C
  # A, B, C: matrices

  let
    al = alpha
    be = beta

  check cublasSetStream(defaultHandle, defaultStream)

  when T is float32:
    cublasSgemm(defaultHandle,
                transa, transb,
                m.cint, n.cint, k.cint,
                al.unsafeAddr, A, lda.cint,
                B, ldb.cint,
                be.unsafeAddr, C, ldc.cint)
  elif T is float64:
    cublasDgemm(defaultHandle,
                transa, transb,
                m.cint, n.cint, k.cint,
                al.unsafeAddr, A, lda.cint,
                B, ldb.cint,
                be.unsafeAddr, C, ldc.cint)
