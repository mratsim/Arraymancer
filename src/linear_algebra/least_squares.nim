# Copyright (c) 2018 the Arraymancer contributors
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

import  ../tensor/tensor, ../tensor/backend/metadataArray,
        ../tensor/private/p_init_cpu,
        nimlapack, fenv, math


proc gelsd(m: ptr cint; n: ptr cint; nrhs: ptr cint; a: ptr cfloat; lda: ptr cint;
            b: ptr cfloat; ldb: ptr cint; s: ptr cfloat; rcond: ptr cfloat; rank: ptr cint;
            work: ptr cfloat; lwork: ptr cint; iwork: ptr cint; info: ptr cint) {.inline.}=
  sgelsd(
    m, n, nrhs,
    a, m,
    b, ldb,
    s, rcond, rank,
    work, lwork,
    iwork, info
  )

proc gelsd(m: ptr cint; n: ptr cint; nrhs: ptr cint; a: ptr cdouble; lda: ptr cint;
            b: ptr cdouble; ldb: ptr cint; s: ptr cdouble; rcond: ptr cdouble;
            rank: ptr cint; work: ptr cdouble; lwork: ptr cint; iwork: ptr cint;
            info: ptr cint) {.inline.}=
  dgelsd(
    m, n, nrhs,
    a, m,
    b, ldb,
    s, rcond, rank,
    work, lwork,
    iwork, info
  )


proc least_squares_solver*[T: SOmeReal](a, b: Tensor[T]):
  tuple[
    least_square_sol: Tensor[T],
    residuals:  Tensor[T],
    matrix_rank: int,
    singular_values: Tensor[T]
  ] {.noInit.}=

  assert a.rank == 2 and b.rank in {1, 2} and a.shape[0] > 0 and a.shape[0] == b.shape[0]
  # We need to copy the input as Lapack will destroy A and replace B with its result.
  # Furthermore it expects a column major ordering.
  # In the future with move operator we can consider letting moved tensors be destroyed.

  # Note on dealing with row-major without copies:
  # http://drsfenner.org/blog/2016/01/into-the-weeds-iii-hacking-lapack-calls-to-save-memory/

  # LAPACK reference doc:
  # http://www.netlib.org/lapack/explore-html/d7/d3b/group__double_g_esolve_ga94bd4a63a6dacf523e25ff617719f752.html
  # Intel example for the Fortran API
  # https://software.intel.com/sites/products/documentation/doclib/mkl_sa/11/mkl_lapack_examples/dgelsd_ex.c.htm

  let is1d = b.rank == 1 # Axis 1 will be squeezed at the end for the 1d case

  var # Parameters
    m = a.shape[0].cint
    n = a.shape[1].cint
    nrhs = if is1d: 1.cint
           else: b.shape[1].cint
    # A is row-major so lda = m
    ldb = max(m, n).cint

  # Shadowing the inputs
  var a = a.clone(colMajor) # Lapack destroys the A matrix during solving

  # Lapack replaces the B matrix by the result during solving
  # Furthermore, as we have input B shape M x NRHS and output N x NRHS
  # if M < N we must zero the reminder of the tensor
  var bstar: Tensor[T]
  tensorCpu([ldb.int, nrhs.int], bstar, colMajor)
  bstar.storage.Fdata = newSeq[T](bstar.size)

  var bstar_slice = bstar[0 ..< b.shape[0], 0 ..< nrhs] # Workaround because slicing does no produce a var at the moment
  apply2_inline(bstar_slice, b):
    # paste b in bstar.
    # if bstar is larger, the rest is zeros
    y

  # Temporary sizes
  const smlsiz = 25.cint # equal to the maximum size of the subproblems at the bottom of the computation tree
  let
    minmn = min(m,n).int
    nlvl = max(0, (minmn.T / (smlsiz+1).T).ln.cint + 1)
    # Bug in some Lapack, liwork must be determined manually: http://icl.cs.utk.edu/lapack-forum/archives/lapack/msg00899.html
    liwork = max(1, 3 * minmn * nlvl + 11 * minmn)

  result.singular_values = newTensorUninit[T](minmn) # will hold the singular values of A

  var # Temporary parameter values
    # Condition for a float to be considered 0
    rcond = epsilon(T) * a.shape.max.T * a.max
    lwork = max(1, 12 * m + 2 * m * smlsiz + 8 * m * nlvl + m * nrhs + (smlsiz + 1) ^ 2)
    work = newSeqUninitialized[T](lwork)
    iwork = newSeqUninitialized[cint](liwork)
    info, rank: cint

  # Solve the equations A*X = B
  gelsd(
    m.addr, n.addr, nrhs.addr,
    a.get_data_ptr, m.addr, # lda
    bstar.get_data_ptr, ldb.addr,
    result.singular_values[0].addr, rcond.addr, rank.addr,
    work[0].addr, lwork.addr,
    iwork[0].addr, info.addr
  )

  if info > 0:
    # TODO, this should not be an exception, not converging is something that can happen and should
    # not interrupt the program. Alternative. Fill the result with Inf?
    raise newException(ValueError, "the algorithm for computing the SVD failed to converge")

  if info < 0:
    raise newException(ValueError, "Illegal parameter in linear square solver gelsd: " & $(-info))

  result.matrix_rank = rank.int # This is the matrix_rank not the tensor rank
  result.least_square_sol = bstar[0 ..< n, _].squeeze(axis = 1) # Correction for 1d case

  if rank == n and m > n:
    result.residuals = (bstar[n .. _, _].squeeze(axis = 1)).fold_axis_inline(Tensor[T], fold_axis = 0):
      x = y .^ 2f # initial value
    do:
      x += y .^ 2f # core loop
    do:
      x += y # values were stored in a temporary array of size == nb of cores
              # to avoid multithreading issues and must be reduced one last time
  else:
    result.residuals = newTensorUninit[T](0)
    # Workaround as we can't create empty tensors for now:
    result.residuals.shape[0] = 0
    result.residuals.shape.len = 0
    result.residuals.strides[0] = 0
    result.residuals.shape.len = 0
