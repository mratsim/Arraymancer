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

proc bmmImpl[T](alpha: T, a, b: Tensor[T], beta: T): Tensor[T] {.noinit.} =
  when compileOption("boundChecks"):
    if a.rank < 2 or b.rank < 2:
      raise newException(ValueError, "Batch matrix multiplication requires both tensors to have rank >= 2, got rank " & $a.rank & " and " & $b.rank)

  let M = a.shape[a.rank - 2]
  let Ka = a.shape[a.rank - 1]
  let Kb = b.shape[b.rank - 2]
  let N = b.shape[b.rank - 1]

  when compileOption("boundChecks"):
    if Ka != Kb:
      raise newException(ValueError, "Matrix inner dimensions must agree for multiplication: " & $Ka & " vs " & $Kb)

  let a_batch_rank = a.rank - 2
  let b_batch_rank = b.rank - 2
  let out_batch_rank = max(a_batch_rank, b_batch_rank)

  var out_batch_shape = newSeq[int](out_batch_rank)
  for i in 0 ..< out_batch_rank:
    let a_dim = if i < out_batch_rank - a_batch_rank: 1 else: a.shape[i - (out_batch_rank - a_batch_rank)]
    let b_dim = if i < out_batch_rank - b_batch_rank: 1 else: b.shape[i - (out_batch_rank - b_batch_rank)]
    if a_dim == b_dim:
      out_batch_shape[i] = a_dim
    elif a_dim == 1:
      out_batch_shape[i] = b_dim
    elif b_dim == 1:
      out_batch_shape[i] = a_dim
    else:
      raise newException(ValueError, "Batch dimension mismatch in matrix multiplication: " & $a.shape & " vs " & $b.shape)

  var out_shape = out_batch_shape
  out_shape.add(M)
  out_shape.add(N)

  result = newTensorUninit[T](out_shape)

  var total_batches = 1
  for d in out_batch_shape:
    total_batches *= d

  for b_idx in 0 ..< total_batches:
    var curr = b_idx
    var a_off = a.offset
    var b_off = b.offset
    var res_off = result.offset

    for i in countdown(out_batch_rank - 1, 0):
      let coord = curr mod out_batch_shape[i]
      curr = curr div out_batch_shape[i]

      res_off += coord * result.strides[i]

      let a_idx = i - (out_batch_rank - a_batch_rank)
      if a_idx >= 0:
        let a_coord = if a.shape[a_idx] == 1: 0 else: coord
        a_off += a_coord * a.strides[a_idx]

      let b_idx_pos = i - (out_batch_rank - b_batch_rank)
      if b_idx_pos >= 0:
        let b_coord = if b.shape[b_idx_pos] == 1: 0 else: coord
        b_off += b_coord * b.strides[b_idx_pos]

    var a_slice: Tensor[T]
    a_slice.shape = [M, Ka].toMetadata
    a_slice.strides = [a.strides[a.rank - 2], a.strides[a.rank - 1]].toMetadata
    a_slice.offset = a_off
    a_slice.storage = a.storage

    var b_slice: Tensor[T]
    b_slice.shape = [Kb, N].toMetadata
    b_slice.strides = [b.strides[b.rank - 2], b.strides[b.rank - 1]].toMetadata
    b_slice.offset = b_off
    b_slice.storage = b.storage

    var res_slice: Tensor[T]
    res_slice.shape = [M, N].toMetadata
    res_slice.strides = [result.strides[result.rank - 2], result.strides[result.rank - 1]].toMetadata
    res_slice.offset = res_off
    res_slice.storage = result.storage

    gemm(alpha, a_slice, b_slice, beta, res_slice)

proc `*`*[T: SomeNumber](a, b: Tensor[T]): Tensor[T] {.noinit.} =
  ## Matrix multiplication (Matrix-Matrix, Matrix-Vector, and Batched Matrix Multiplication)
  ##
  ## Float and complex operations use optimized BLAS like OpenBLAS, Intel MKL or BLIS.

  if a.rank == 2 and b.rank == 2:
    result = newTensorUninit[T](a.shape[0], b.shape[1])
    gemm(1.T, a, b, 0.T, result)
  elif a.rank == 2 and b.rank == 1:
    result = newTensorUninit[T](a.shape[0])
    gemv(1.T, a, b, 0.T, result)
  elif a.rank >= 2 and b.rank >= 2:
    result = bmmImpl(1.T, a, b, 0.T)
  else:
    raise newException(ValueError, "Matrix multiplication valid only if both tensors have rank >= 2 (batched matrix multiplication), or rank 2 and rank 1 (matrix-vector)")

proc `*`*[T: Complex[float32] or Complex[float64]](
      a, b: Tensor[T]): Tensor[T] {.noinit.} =
  ## Matrix multiplication (Matrix-Matrix, Matrix-Vector, and Batched Matrix Multiplication)
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
  elif a.rank >= 2 and b.rank >= 2:
    result = bmmImpl(complex(1.F, 0.F), a, b, complex(0.F, 0.F))
  else:
    raise newException(ValueError, "Matrix multiplication valid only if both tensors have rank >= 2 (batched matrix multiplication), or rank 2 and rank 1 (matrix-vector)")


