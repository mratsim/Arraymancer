# Copyright (c) 2018 Mamy André-Ratsimbazafy and the Arraymancer contributors
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

import ../tensor/tensor, nimlapack

# SVD and Eigenvalues/eigenvectors decomposition

proc syev*(jobz: cstring; uplo: cstring; n: ptr cint; a: ptr cfloat; lda: ptr cint;
          w: ptr cfloat; work: ptr cfloat; lwork: ptr cint; info: ptr cint) {.inline.}=

  ssyev(jobz, uplo, n, a, lda,
        w, work, lwork, info)

proc syev*(jobz: cstring; uplo: cstring; n: ptr cint; a: ptr cdouble; lda: ptr cint;
          w: ptr cdouble; work: ptr cdouble; lwork: ptr cint; info: ptr cint) {.inline.}=

  dsyev(jobz, uplo, n, a, lda,
        w, work, lwork, info)

proc symeig*[T: SomeReal](a: Tensor[T]): tuple[eigenval, eigenvec: Tensor[T]] =
  # Compute the eigenvalues and eigen vectors of a symmetric matrix

  assert a.shape[0] == a.shape[1], "Input should be a symmetric matrix"
  # TODO, support "symmetric matrices" with only the upper or lower part filled.
  # (Obviously, upper in Fortran is lower in C ...)

  # input is destroyed by LAPACK
  result.eigenvec = a.clone(layout = colMajor)

  # Locals
  var
    n, lda: cint = a.shape[0].cint
    info: cint
    wkopt: T
    lwork: cint = -1
    jobz: cstring = "V" # N or V (eigenval only or eigenval + eigen vec)
    uplo: cstring = "U" # U or L (upper or lower, in ColMajor layout)

  result.eigenval = newTensorUninit[T](a.shape[0])
  let w = result.eigenval.get_data_ptr
  let vec = result.eigenvec.get_data_ptr

  # Query and allocate optimal workspace
  syev(jobz, uplo, n.addr, vec, lda.addr, w, wkopt.addr, lwork.addr, info.addr)

  lwork = wkopt.cint
  var work = newSeq[T](lwork)

  # Solve eigenproblem
  syev(jobz, uplo, n.addr, vec, lda.addr, w, work[0].addr, lwork.addr, info.addr)

  if info > 0:
    # TODO, this should not be an exception, not converging is something that can happen and should
    # not interrupt the program. Alternative. Fill the result with Inf?
    raise newException(ValueError, "the algorithm for computing the SVD failed to converge")

  if info < 0:
    raise newException(ValueError, "Illegal parameter in linear square solver gelsd: " & $(-info))

when isMainModule:

  let a =  [[1.96, -6.49, -0.47, -7.20, -0.65],
            [0.00,  3.80, -6.39,  1.50, -6.34],
            [0.00,  0.00,  4.17, -1.51,  2.67],
            [0.00,  0.00,  0.00,  5.70,  1.80],
            [0.00,  0.00,  0.00,  0.00, -7.10]].toTensor

  echo symeig(a)
