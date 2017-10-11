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

import  ./private/p_checks,
        ./private/p_operator_blas_l2l3,
        ./fallback/naive_l2_gemv,
        ./data_structure,
        ./init_cpu

# #################################################
# Generic notation "*"

proc `*`*[T: SomeReal](a, b: Tensor[T]): Tensor[T] =
  ## Matrix multiplication (Matrix-Matrix and Matrix-Vector)
  ##
  ## Float operations use optimized BLAS like OpenBLAS, Intel MKL or BLIS.

  when declared(blis):
    ## When is evaluated at compile time and has no runtime cost
    if not a.isContiguous or not b.isContiguous:
      # OpenBLAS / MKL are still faster than BLIS in the contiguous case
      if a.rank == 2 and b.rank == 2:
        when compileOption("boundChecks"):
          check_matmat(a,b)
        result = newTensorUninit[T](a.shape[0], b.shape[1])
        blisMM_C_eq_aAB_p_bC(1.T, a, b, 0.T, result)
      elif a.rank == 2 and b.rank == 1:
        when compileOption("boundChecks"):
          check_matvec(a,b)
        result = newTensorUninit[T](a.shape[0])
        blisMV_y_eq_aAx_p_by(1.T, a, b, 0.T, result)
      else:
        raise newException(ValueError, "Matrix-Matrix or Matrix-Vector multiplication valid only if first Tensor is a Matrix and second is a Matrix or Vector")
      return

  if a.rank == 2 and b.rank == 2:
    when compileOption("boundChecks"):
      check_matmat(a,b)
    result = newTensorUninit[T](a.shape[0], b.shape[1])
    blasMM_C_eq_aAB_p_bC(1.T, a, b, 0.T, result)
  elif a.rank == 2 and b.rank == 1:
    when compileOption("boundChecks"):
      check_matvec(a,b)
    result = newTensorUninit[T](a.shape[0])
    blasMV_y_eq_aAx_p_by(1.T, a, b, 0.T, result)
  else:
    raise newException(ValueError, "Matrix-Matrix or Matrix-Vector multiplication valid only if first Tensor is a Matrix and second is a Matrix or Vector")

proc `*`*[T: SomeInteger](a, b: Tensor[T]): Tensor[T] =
  ## Matrix-Matrix and Matrix-Vector multiplications fallback for integer tensors.
  ##
  ## Integer BLAS has been implemented manually. While not as fast as BLAS for floats,
  ## it should be much faster than naive loops.
  ##
  ## Note: Integers smaller than 2^31 can be converted to float64 without losing precision
  ## and can benefit from the optimized float BLAS implementations

  # static: echo "Please note that integer matrix-matrix and matrix-vector multiplications do not have optimized " &
  #              "operations like how research has done for floats. If your integers are " &
  #              "smaller than 2^31, you can convert them to float64 without losing precision before " &
  #              "Matrix-Matrix or Matrix-Vector operations to benefit from accelerated routines."

  if a.rank == 2 and b.rank == 2:
    when compileOption("boundChecks"):
      check_matmat(a,b)
    result = newTensorUninit[T](a.shape[0], b.shape[1])
    fallbackMM_C_eq_aAB_p_bC(1.T, a, b, 0.T, result)
  elif a.rank == 2 and b.rank == 1:
    when compileOption("boundChecks"):
      check_matvec(a,b)
    result = newTensorUninit[T](a.shape[0])
    naive_gemv_fallback(1.T, a, b, 0.T, result)
  else:
    raise newException(ValueError, "Matrix-Matrix or Matrix-Vector multiplication valid only if first Tensor is a Matrix and second is a Matrix or Vector")