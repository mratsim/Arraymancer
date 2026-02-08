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

import  ./backend/metal/metal_backend,
        ./private/p_init_metal,
        ./private/p_checks_metal,
        ./data_structure,
        ./shapeshifting_metal

proc metalMV_y_eq_aAx_p_by[T: SomeFloat](
  alpha: T, a, x: MetalTensor[T],
  beta: T, y: var MetalTensor[T]) =
  # Matrix-Vector: y = alpha A matvecmul x + beta y

  # TODO: remove this contiguous layout constraint
  if not a.isContiguous:
    raise newException(ValueError, "NotImplemented: for now matrix should be contiguous")

  let M = a.shape[0]
  let N = a.shape[1]
  
  # Use GPU kernel for matrix-vector multiplication
  metalGemv[T](
    false,  # trans
    M, N,
    alpha,
    a.storage.Fbuffer, M,  # lda = rows of A
    x.storage.Fbuffer, 1,  # incx = 1
    beta,
    y.storage.Fbuffer, 1   # incy = 1
  )

proc metalMM_C_eq_aAB_p_bC[T: SomeFloat](
  alpha: T, a, b: MetalTensor[T],
  beta: T, c: var MetalTensor[T]) =
  # Matrix: C = alpha A matmul B + beta C
  # All tensors are in column-major order (Fortran style)

  # TODO: remove this contiguous layout constraint
  if not (a.isContiguous and b.isContiguous):
    raise newException(ValueError, "NotImplemented: for now both tensors should be contiguous")

  let M = a.shape[0]  # rows of A and C
  let N = b.shape[1]  # cols of B and C
  let K = a.shape[1]  # cols of A, rows of B
  
  # For column-major matrices:
  # lda = number of rows in A (M)
  # ldb = number of rows in B (K)
  # ldc = number of rows in C (M)
  metalGemm[T](
    false, false,  # transA, transB
    M, N, K,
    alpha,
    a.storage.Fbuffer, M,  # lda = rows of A
    b.storage.Fbuffer, K,  # ldb = rows of B
    beta,
    c.storage.Fbuffer, M   # ldc = rows of C
  )

proc `*`*[T: SomeFloat](a, b: MetalTensor[T]): MetalTensor[T] =
  ## Matrix multiplication (Matrix-Matrix and Matrix-Vector) on Metal

  if a.rank == 2 and b.rank == 2:
    when compileOption("boundChecks"):
      check_matmat(a, b)
    result = newMetalTensor[T]([a.shape[0], b.shape[1]])
    metalMM_C_eq_aAB_p_bC(1.T, a, b, 0.T, result)
  elif a.rank == 2 and b.rank == 1:
    when compileOption("boundChecks"):
      check_matvec(a, b)
    result = newMetalTensor[T]([a.shape[0]])
    metalMV_y_eq_aAx_p_by(1.T, a, b, 0.T, result)
  else:
    raise newException(ValueError, "Matrix-Matrix or Matrix-Vector multiplication valid only if first Tensor is a Matrix and second is a Matrix or Vector")
