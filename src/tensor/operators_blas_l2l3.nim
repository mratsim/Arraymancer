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
  import ./backend/blis

import  ./private/p_checks,
        ./private/p_operator_blas_l2l3,
        ./fallback/naive_l2_gemv,
        ./data_structure,
        ./init_cpu

proc gemv*[T: SomeFloat|Complex](
          alpha: T,
          A: Tensor[T],
          x: Tensor[T],
          beta: T,
          y: var Tensor[T]) {.inline.}=
  ## General Matrix-Vector multiplication:
  ## y <- alpha * A * x + beta * y
  when compileOption("boundChecks"):
    check_matvec(A,x)
    # TODO: check y + tests
  when declared(blis):
    # OpenBLAS / MKL are still faster than BLIS in the contiguous case
    # For matrix vector, the vector can be non-contiguous for MKL / OpenBLAS
    if not A.isContiguous:
      blisMV_y_eq_aAx_p_by(alpha, A, x, beta, y)
      return

  blasMV_y_eq_aAx_p_by(alpha, A, x, beta, y)

proc gemv*[T: SomeInteger](
          alpha: T,
          A: Tensor[T],
          x: Tensor[T],
          beta: T,
          y: var Tensor[T]) {.inline.}=
  ## General Matrix-Vector multiplication:
  ## y <- alpha * A * x + beta * y
  when compileOption("boundChecks"):
    check_matvec(A,x)
    # TODO: check y + tests

  naive_gemv_fallback(alpha, A, x, beta, y)

proc gemm*[T: SomeFloat|Complex](
  alpha: T, A, B: Tensor[T],
  beta: T, C: var Tensor[T]) {.inline.}=
  # Matrix: C = alpha A matmul B + beta C
  when compileOption("boundChecks"):
    check_matmat(A,B)
    # TODO: check c + tests

  when declared(blis):
    if not A.isContiguous or not B.isContiguous or not C.isContiguous:
      blisMM_C_eq_aAB_p_bC(alpha, A, B, beta, C)
      return

  blasMM_C_eq_aAB_p_bC(alpha, A, B, beta, C)

proc gemm*[T: SomeInteger](
  alpha: T, A, B: Tensor[T],
  beta: T, C: var Tensor[T]) {.inline.}=
  # Matrix: C = alpha A matmul B + beta C
  when compileOption("boundChecks"):
    check_matmat(A,B)
    # TODO: check c + tests

  fallbackMM_C_eq_aAB_p_bC(alpha, A, B, beta, C)

proc gemm*[T: SomeNumber](
  A, B: Tensor[T],
  C: var Tensor[T]) {.deprecated: "Use explicit gemm(1, A, B, 0, C) instead".}=
  gemm(1.T, A, B, 0.T, C)

proc `*`*[T: SomeNumber](a, b: Tensor[T]): Tensor[T] {.noInit.} =
  ## Matrix multiplication (Matrix-Matrix and Matrix-Vector)
  ##
  ## Float and complex operations use optimized BLAS like OpenBLAS, Intel MKL or BLIS.

  if a.rank == 2 and b.rank == 2:
    result = newTensorUninit[T](a.shape[0], b.shape[1])
    gemm(1.T, a, b, 0.T, result)
  elif a.rank == 2 and b.rank == 1:
    result = newTensorUninit[T](a.shape[0])
    gemv(1.T, a, b, 0.T, result)
  else:
    raise newException(ValueError, "Matrix-Matrix or Matrix-Vector multiplication valid only if first Tensor is a Matrix and second is a Matrix or Vector")

proc `*`*[T: Complex[float32] or Complex[float64]](
      a, b: Tensor[T]): Tensor[T] {.noInit.} =
  ## Matrix multiplication (Matrix-Matrix and Matrix-Vector)
  ##
  ## Float and complex operations use optimized BLAS like OpenBLAS, Intel MKL or BLIS.

  type F = T.T # Get float subtype of Complex[T]
  # We need to workaround https://github.com/nim-lang/Nim/issues/12525
  # and not use the default parameter

  if a.rank == 2 and b.rank == 2:
    result = newTensorUninit[T](a.shape[0], b.shape[1])
    gemm(complex(1.F, 0.F), a, b, complex(0.F, 0.F), result)
  elif a.rank == 2 and b.rank == 1:
    result = newTensorUninit[T](a.shape[0])
    gemv(complex(1.F, 0.F), a, b, complex(0.F, 0.F), result)
  else:
    raise newException(ValueError, "Matrix-Matrix or Matrix-Vector multiplication valid only if first Tensor is a Matrix and second is a Matrix or Vector")
