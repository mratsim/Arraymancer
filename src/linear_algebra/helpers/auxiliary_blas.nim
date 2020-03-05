# Copyright (c) 2018-Present Mamy André-Ratsimbazafy
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

import
  nimblas,
  ../../tensor/tensor


# Auxiliary functions from BLAS
# ----------------------------------

type
  SyrkKind* = enum
    AAt = "A * A.transpose"
    AtA = "A.transpose * A"

proc syrk*[T: SomeFloat](
    alpha: T, A: Tensor[T],
    mul_order: static SyrkKind,
    beta: T, C: var Tensor[T], uplo: static char) =

  static: assert uplo in {'U','L'}
  const uplo = if uplo == 'U': upper
               else: lower

  let N = C.shape[0].int32
  let K = if mul_order == AAt: A.shape[1].int32
          else: A.shape[0].int32
  assert C.rank == 2, "C must be a matrix"
  assert N == C.shape[1].int32, "C must be a square matrix"
  when compileOption("boundChecks"):
    if mul_order == AAt:
      assert N == A.shape[0]
    else:
      assert N == A.shape[1]

  var layout: nimblas.OrderType
  var trans: nimblas.TransposeType
  var lda: int32

  # Someone saves me from BLAS rowMajor/colMajor/lda/transpose ceremony
  if C.is_F_contiguous:
    layout = colMajor
    if A.is_F_contiguous:
      when mul_order == AAt:
        trans = noTranspose
        lda = N
      else:
        trans = transpose
        lda = K
    elif A.is_C_contiguous:
      when mul_order == AAt:
        trans = transpose
        lda = K
      else:
        trans = noTranspose
        lda = N
    else:
      raise newException(ValueError, "A must be contiguous")
  elif C.is_C_contiguous:
    layout = rowMajor
    if A.is_C_contiguous:
      when mul_order == AAt:
        trans = noTranspose
        lda = K
      else:
        trans = transpose
        lda = N
    elif A.is_F_contiguous:
      when mul_order == AAt:
        trans = transpose
        lda = N
      else:
        trans = noTranspose
        lda = K
    else:
      raise newException(ValueError, "A must be contiguous")
  else:
    raise newException(ValueError, "C must be contiguous")
  # And done, good luck testing that.

  # The C interface to BLAS will do the correct thing
  # regarding to uplo and the requested layout of C
  syrk(layout, uplo, trans, N, K,
       alpha, A.get_data_ptr, lda,
       beta, C.get_data_ptr, N
      )

# Sanity checks
# ----------------------------------

proc pttr(a: var Tensor, uplo: static char) =
  # Convert a symmetric matrix from packed to full storage

  assert a.rank == 2
  let N = a.shape[0]
  assert N == a.shape[1]

  let A = a.unsafe_raw_offset()
  when uplo == 'U':
    if a.is_F_contiguous:
      for i in 0 ..< N:
        for j in i+1 ..< N:
          A[i*N + j] = A[j*N + i]
    else:
      for i in 0 ..< N:
        for j in i+1 ..< N:
          A[j*N + i] = A[i*N + j]
  else:
    {.error: "Not implemented".}


when isMainModule:
  import ./init_colmajor

  let A = [[1.0, 2],
          [3.0, 4],
          [5.0, 6]]
  var C: Tensor[float64]

  let aat = A.toTensor() * A.toTensor().transpose()
  let ata = A.toTensor().transpose() * A.toTensor()

  block: # C col-major, A col-major, A*A.t
    let A = A.toTensor.asContiguous(colMajor, force = true)
    C.newMatrixUninitColMajor(3, 3)
    syrk(1.0, A, AAt, 0, C, 'U')
    pttr(C, 'U')
    doAssert C == aat
  block: # C col-major, A col-major, A.t*A
    let A = A.toTensor.asContiguous(colMajor, force = true)
    C.newMatrixUninitColMajor(2, 2)
    syrk(1.0, A, AtA, 0, C, 'U')
    pttr(C, 'U')
    doAssert C == ata
  block: # C col-major, A row-major, A*A.t
    let A = A.toTensor
    C.newMatrixUninitColMajor(3, 3)
    syrk(1.0, A, AAt, 0, C, 'U')
    pttr(C, 'U')
    doAssert C == aat
  block: # C col-major, A row-major, A.t*A
    let A = A.toTensor
    C.newMatrixUninitColMajor(2, 2)
    syrk(1.0, A, AtA, 0, C, 'U')
    pttr(C, 'U')
    doAssert C == ata
  block: # C row-major, A col-major, A*A.t
    let A = A.toTensor.asContiguous(colMajor, force = true)
    C = newTensorUninit[float64](3, 3)
    syrk(1.0, A, AAt, 0, C, 'U')
    pttr(C, 'U')
    doAssert C == aat
  block: # C row-major, A col-major, A.t*A
    let A = A.toTensor.asContiguous(colMajor, force = true)
    C = newTensorUninit[float64](2, 2)
    syrk(1.0, A, AtA, 0, C, 'U')
    pttr(C, 'U')
    doAssert C == ata
  block: # C row-major, A row-major, A*A.t
    let A = A.toTensor
    C = newTensorUninit[float64](3, 3)
    syrk(1.0, A, AAt, 0, C, 'U')
    pttr(C, 'U')
    doAssert C == aat
  block: # C row-major, A row-major, A.t*A
    let A = A.toTensor
    C = newTensorUninit[float64](2, 2)
    syrk(1.0, A, AtA, 0, C, 'U')
    pttr(C, 'U')
    doAssert C == ata
