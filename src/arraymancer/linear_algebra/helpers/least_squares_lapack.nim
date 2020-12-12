# Copyright (c) 2018-Present Mamy André-Ratsimbazafy
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

import
  nimlapack, fenv,
  ./overload, ./init_colmajor,
  ../../private/sequninit,
  ../../tensor

# Least Squares using Recursive Divide & Conquer
# --------------------------------------------------------------------------------------

overload(gelsd, sgelsd)
overload(gelsd, dgelsd)

proc gelsd*[T: SomeFloat](
      a, b: Tensor[T],
      solution, residuals: var Tensor[T],
      singular_values: var Tensor[T],
      matrix_rank: var int
    ) =

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
    m = a.shape[0].int32
    n = a.shape[1].int32
    nrhs = if is1d: 1.int32
           else: b.shape[1].int32
    # A is row-major so lda = m
    ldb = max(m, n).int32

  # Shadowing the inputs
  var a = a.clone(colMajor) # Lapack destroys the A matrix during solving

  # Lapack replaces the B matrix by the result during solving
  # Furthermore, as we have input B shape M x NRHS and output N x NRHS
  # if M < N we must zero the remainder of the tensor
  var b2: Tensor[T]
  b2.newMatrixUninitColMajor(ldb.int, nrhs.int)

  var b2_slice = b2[0 ..< b.shape[0], 0 ..< nrhs]
  if is1d:
    b2_slice = b2_slice.squeeze(1)

  forEach ib2 in b2_slice,
          ib in b:
    ib2 = ib

  # Temporary sizes
  const smlsiz = 25'i32 # equal to the maximum size of the subproblems at the bottom of the computation tree
  let
    minmn = min(m,n).int
    nlvl = max(0, (minmn.T / (smlsiz+1).T).ln.int32 + 1)
    # Bug in some Lapack, liwork must be determined manually: http://icl.cs.utk.edu/lapack-forum/archives/lapack/msg00899.html
    liwork = max(1, 3 * minmn * nlvl + 11 * minmn)

  singular_values = newTensorUninit[T](minmn) # will hold the singular values of A

  var # Temporary parameter values
    # Condition for a float to be considered 0
    rcond = epsilon(T) * a.shape.max.T * a.max
    lwork = max(1, 12 * m + 2 * m * smlsiz + 8 * m * nlvl + m * nrhs + (smlsiz + 1) ^ 2)
    work = newSeqUninit[T](lwork)
    iwork = newSeqUninit[cint](liwork)
    info, rank: int32

  # Solve the equations A*X = B
  gelsd(
    m.addr, n.addr, nrhs.addr,
    a.get_data_ptr, m.addr, # lda
    b2.get_data_ptr, ldb.addr,
    singular_values[0].addr, rcond.addr, rank.addr,
    work[0].addr, lwork.addr,
    iwork[0].addr, info.addr
  )

  if info > 0:
    # TODO, this should not be an exception, not converging is something that can happen and should
    # not interrupt the program. Alternative. Fill the result with Inf?
    # Though scipy / sklearn seems to use LinAlgError exceptions
    raise newException(ValueError, "the algorithm for computing the SVD failed to converge")

  if info < 0:
    raise newException(ValueError, "Illegal parameter in linear square solver gelsd: " & $(-info))

  matrix_rank = rank.int # This is the matrix_rank not the tensor rank
  solution = b2[0 ..< n, _].squeeze(axis = 1) # Correction for 1d case

  if rank == n and m > n:
    residuals = (b2[n .. _, _].squeeze(axis = 1)).fold_axis_inline(Tensor[T], fold_axis = 0):
      x = y ^. 2f # initial value
    do:
      x += y ^. 2f # core loop
    do:
      x += y # values were stored in a temporary array of size == nb of cores
              # to avoid multithreading issues and must be reduced one last time
  else:
    residuals = newTensorUninit[T](0)
    # Workaround as we can't create empty tensors for now:
    residuals.shape[0] = 0
    residuals.shape.len = 0
    residuals.strides[0] = 0
    residuals.shape.len = 0
