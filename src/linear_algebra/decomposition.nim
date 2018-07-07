# Copyright (c) 2018 Mamy André-Ratsimbazafy and the Arraymancer contributors
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

import
  ../private/sequninit,
  ../tensor/tensor, nimlapack,
  ../tensor/private/p_init_cpu # TODO: can't call newTensorUninit with optional colMajor, varargs breaks it

# SVD and Eigenvalues/eigenvectors decomposition

proc syevr(jobz: cstring; range: cstring; uplo: cstring; n: ptr cint; a: ptr cfloat;
            lda: ptr cint; vl: ptr cfloat; vu: ptr cfloat; il: ptr cint; iu: ptr cint;
            abstol: ptr cfloat; m: ptr cint; w: ptr cfloat; z: ptr cfloat; ldz: ptr cint;
            isuppz: ptr cint; work: ptr cfloat; lwork: ptr cint; iwork: ptr cint;
            liwork: ptr cint; info: ptr cint) {.inline.} =

  ssyevr(jobz, range, uplo, n, a,
          lda, vl, vu, il, iu,
          abstol, m, w, z, ldz,
          isuppz, work, lwork, iwork,
          liwork, info)

proc syevr(jobz: cstring; range: cstring; uplo: cstring; n: ptr cint; a: ptr cdouble;
            lda: ptr cint; vl: ptr cdouble; vu: ptr cdouble; il: ptr cint; iu: ptr cint;
            abstol: ptr cdouble; m: ptr cint; w: ptr cdouble; z: ptr cdouble; ldz: ptr cint;
            isuppz: ptr cint; work: ptr cdouble; lwork: ptr cint; iwork: ptr cint;
            liwork: ptr cint; info: ptr cint) {.inline.} =

  dsyevr(jobz, range, uplo, n, a,
          lda, vl, vu, il, iu,
          abstol, m, w, z, ldz,
          isuppz, work, lwork, iwork,
          liwork, info)


proc symeigImpl[T: SomeFloat](a: Tensor[T], eigenvectors: bool,
  low_idx: int, high_idx: int, result: var tuple[eigenval, eigenvec: Tensor[T]]) =

  assert a.shape[0] == a.shape[1], "Input should be a symmetric matrix"
  # TODO, support "symmetric matrices" with only the upper or lower part filled.
  # (Obviously, upper in Fortran is lower in C ...)

  let a = a.clone(colMajor) # Lapack overwrite the input. TODO move optimization

  var
    jobz: cstring
    interval: cstring
    n, lda: cint = a.shape[0].cint
    uplo: cstring = "U"
    vl, vu: T         # unused: min and max eigenvalue returned
    il, iu: cint      # ranking of the lowest and highest eigenvalues returned
    abstol: T = -1    # Use default. otherwise need to call LAPACK routine dlamch('S') or 2*dlamch('S')
    m, ldz = n
    lwork: cint = -1  # dimension of a workspace array
    work: seq[T]
    wkopt: T
    liwork: cint = -1 # dimension of a second workspace array
    iwork: seq[cint]
    iwkopt: cint
    info: cint

  if low_idx == 0 and high_idx == a.shape[0] - 1:
    interval = "A"
  else:
    interval = "I"
    il = cint low_idx + 1 # Fortran starts indexing with 1
    iu = cint high_idx + 1
    m = iu - il + 1

  # Setting up output
  var
    isuppz: seq[cint] # unused
    isuppz_ptr: ptr cint
    z: ptr T

  result.eigenval = newTensorUninit[T](a.shape[0]) # Even if less eigenval are selected Lapack requires this much workspace

  if eigenvectors:
    jobz = "V"

    # Varargs + optional colMajor argument issue, must resort to low level proc at the moment
    tensorCpu(ldz.int, m.int, result.eigenvec, colMajor)
    result.eigenvec.storage.Fdata = newSeqUninit[T](result.eigenvec.size)

    z = result.eigenvec.get_data_ptr
    if interval == "A": # or (il == 1 and iu == n): -> Already checked before
      isuppz = newSeqUninit[cint](2*m)
      isuppz_ptr = isuppz[0].addr
  else:
    jobz = "N"

  let w = result.eigenval.get_data_ptr

  # Querying workspaces sizes
  syevr(jobz, interval, uplo, n.addr, a.get_data_ptr, lda.addr, vl.addr, vu.addr, il.addr, iu.addr,
        abstol.addr, m.addr, w, z, ldz.addr, isuppz_ptr, wkopt.addr, lwork.addr, iwkopt.addr, liwork.addr, info.addr)

  # Allocating workspace
  lwork = wkopt.cint
  work = newSeqUninit[T](lwork)
  liwork = iwkopt.cint
  iwork = newSeqUninit[cint](liwork)

  syevr(jobz, interval, uplo, n.addr, a.get_data_ptr, lda.addr, vl.addr, vu.addr, il.addr, iu.addr,
        abstol.addr, m.addr, w, z, ldz.addr, isuppz_ptr, work[0].addr, lwork.addr, iwork[0].addr, liwork.addr, info.addr)

  # Keep only the selected eigenvals
  result.eigenval = result.eigenval[0 ..< m.int]

  when compileOption("boundChecks"):
    if eigenvectors:
      assert m.int == result.eigenvec.shape[1]

  if unlikely(info > 0):
    # TODO, this should not be an exception, not converging is something that can happen and should
    # not interrupt the program. Alternative. Fill the result with Inf?
    raise newException(ValueError, "the algorithm for computing eigenvalues failed to converge")

  if unlikely(info < 0):
    raise newException(ValueError, "Illegal parameter in symeig: " & $(-info))


template `^^`(s, i: untyped): untyped =
  (when i is BackwardsIndex: s.shape[0] - int(i) else: int(i))

proc symeig*[T: SomeFloat](a: Tensor[T], eigenvectors = false): tuple[eigenval, eigenvec: Tensor[T]] {.inline.}=
  ## Compute the eigenvalues and eigen vectors of a symmetric matrix
  ## Input:
  ##   - A symmetric matrix of shape [n x n]
  ##   - A boolean: true if you also want the eigenvectors, false otherwise
  ## Returns:
  ##   - A tuple with:
  ##     - The eigenvalues sorted from lowest to highest. (shape [n])
  ##     - The corresponding eigenvectors of shape [n, n] if it was requested.
  ##       If eigenvectors were not requested, this have to be discarded.
  ##       Using the result will create a runtime error.
  ##
  ## Implementation is done through the Multiple Relatively Robust Representations

  symeigImpl(a, eigenvectors, 0, a.shape[0] - 1, result)

proc symeig*[T: SomeFloat](a: Tensor[T], eigenvectors = false,
  slice: HSlice[int or BackwardsIndex, int or BackwardsIndex]): tuple[eigenval, eigenvec: Tensor[T]] {.inline.}=
  ## Compute the eigenvalues and eigen vectors of a symmetric matrix
  ## Input:
  ##   - A symmetric matrix of shape [n, n]
  ##   - A boolean: true if you also want the eigenvectors, false otherwise
  ##   - A slice of the rankings of eigenvalues you request. For example requesting
  ##     eigenvalues 2 and 3 would be done with 1..2.
  ## Returns:
  ##   - A tuple with:
  ##     - The eigenvalues sorted from lowest to highest. (shape [m] where m is the slice size)
  ##     - The corresponding eigenvector if it was requested. (shape [n, m])
  ##       If eigenvectors were not requested, this have to be discarded.
  ##       Using the result will create a runtime error.
  ##
  ## Implementation is done through the Multiple Relatively Robust Representations

  symeigImpl(a, eigenvectors, a ^^ slice.a, a ^^ slice.b, result)
