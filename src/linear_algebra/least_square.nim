# Copyright (c) 2018 the Arraymancer contributors
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).

# Note: Lapack-lite size autodetection bug:
# http://icl.cs.utk.edu/lapack-forum/archives/lapack/msg00899.html

import  ../tensor/tensor, ../tensor/backend/metadataArray,
        nimlapack, fenv, math


proc least_square_solver*(a, b: Tensor[float32]): Tensor[float32] =

  assert a.rank == 2 and b.rank in {1, 2} and a.shape[0] > 0 and b.shape[0] > 0
  # We need to copy the input as Lapack will destroy A and replace B with its result.
  # Furthermore it expects a column major ordering.
  # In the future with move operator we can consider letting moved tensors be destroyed.

  # Note on dealing with row-major without copies:
  # http://drsfenner.org/blog/2016/01/into-the-weeds-iii-hacking-lapack-calls-to-save-memory/

  # LAPACK reference doc:
  # http://www.netlib.org/lapack/explore-html/d7/d3b/group__double_g_esolve_ga94bd4a63a6dacf523e25ff617719f752.html
  # Intel example for the Fortran API
  # https://software.intel.com/sites/products/documentation/doclib/mkl_sa/11/mkl_lapack_examples/dgelsd_ex.c.htm

  var a  = a.clone(colMajor) # Lapack destroys the A matrix during solving
  result = b.clone(colMajor) # Lapack replaces the B matrix by the result during solving

  var # Parameters
    m = a.shape[0].cint
    n = a.shape[1].cint
    nrhs = if result.rank == 1: 1.cint
           else: result.shape[1].cint
    # A is row-major so lda = m
    ldb = max(m, n).cint

  # Temporary sizes
  const smlsiz = 25.cint # equal to the maximum size of the subproblems at the bottom of the computation tree
  let
    minmn = min(m,n)
    nlvl = max(0, (minmn.float32 / (smlsiz+1).float32).ln.cint + 1)
    # Bug in some Lapack, liwork must be determined manually: http://icl.cs.utk.edu/lapack-forum/archives/lapack/msg00899.html
    liwork = max(1, 3 * minmn * nlvl + 11 * minmn)

  var # Temporary parameter values
    # Condition for a float to be considered 0
    rcond = epsilon(float32) * a.shape.max.float32 * a.max
    lwork = max(1, 12 * m + 2 * m * smlsiz + 8 * m * nlvl + m * nrhs + (smlsiz + 1) ^ 2)
    work = newSeqUninitialized[float32](lwork)
    iwork = newSeqUninitialized[cint](liwork)
    info, rank: cint
    S = newSeqUninitialized[float32](minmn) # will hold the singular values of A

  # Solve the equations A*X = B
  sgelsd(
    m.addr, n.addr, nrhs.addr,
    a.get_data_ptr, m.addr, # lda
    result.get_data_ptr, ldb.addr,
    S[0].addr, rcond.addr, rank.addr,
    work[0].addr, lwork.addr,
    iwork[0].addr, info.addr
  )

  if info > 0:
    # TODO, this should not be an exception, not converging is something that can happen and should
    # not interrupt the program. Alternative. Fill the result with Inf?
    raise newException(ValueError, "the algorithm for computing the SVD failed to converge")

  if info < 0:
    raise newException(ValueError, "Illegal parameter in linear square solver gelsd: " & $(-info))
