# Copyright (c) 2018 the Arraymancer contributors
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).

# Note: Lapack-lite size autodetection bug:
# http://icl.cs.utk.edu/lapack-forum/archives/lapack/msg00899.html

import  ../tensor/tensor, ../tensor/backend/metadataArray,
        nimlapack, fenv, math


proc least_square_solver*(a, b: Tensor[float32]):
  tuple[
    least_square_sol: Tensor[float32],
    residuals:  Tensor[float32],
    matrix_rank: int,
    singular_values: Tensor[float32]
  ] {.noInit.}=

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

  var a = a.clone(colMajor) # Lapack destroys the A matrix during solving
  var b = b.clone(colMajor) # Lapack replaces the B matrix by the result during solving

  let is1d = b.rank == 1

  var # Parameters
    m = a.shape[0].cint
    n = a.shape[1].cint
    nrhs = if is1d: 1.cint
           else: b.shape[1].cint
    # A is row-major so lda = m
    ldb = max(m, n).cint

  # Temporary sizes
  const smlsiz = 25.cint # equal to the maximum size of the subproblems at the bottom of the computation tree
  let
    minmn = min(m,n).int
    nlvl = max(0, (minmn.float32 / (smlsiz+1).float32).ln.cint + 1)
    # Bug in some Lapack, liwork must be determined manually: http://icl.cs.utk.edu/lapack-forum/archives/lapack/msg00899.html
    liwork = max(1, 3 * minmn * nlvl + 11 * minmn)

  result.singular_values = newTensorUninit[float32](minmn) # will hold the singular values of A

  var # Temporary parameter values
    # Condition for a float to be considered 0
    rcond = epsilon(float32) * a.shape.max.float32 * a.max
    lwork = max(1, 12 * m + 2 * m * smlsiz + 8 * m * nlvl + m * nrhs + (smlsiz + 1) ^ 2)
    work = newSeqUninitialized[float32](lwork)
    iwork = newSeqUninitialized[cint](liwork)
    info, rank: cint

  # Solve the equations A*X = B
  sgelsd(
    m.addr, n.addr, nrhs.addr,
    a.get_data_ptr, m.addr, # lda
    b.get_data_ptr, ldb.addr,
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
  if is1d:
    result.least_square_sol = b[0 ..< n]
    if rank == n and m > n:
      result.residuals = (b[n .. _]).reduce_axis_inline(0):
        x += y .^ 2f
  else:
    result.least_square_sol = b[0 ..< n, _]
    if rank == n and m > n:
      debugecho b
      result.residuals = (b[n .. _, _]).reduce_axis_inline(0):
        x += y .^ 2f


when isMainModule:

  # let a = [
  #   [ 0.12f, -6.91, -3.33,  3.97],
  #   [-8.19f,  2.22, -8.94,  3.33],
  #   [ 7.69f, -5.12, -6.72, -2.74],
  #   [-2.26f, -9.08, -4.40, -7.92],
  #   [-4.71f,  9.96, -9.98, -3.20]
  # ].toTensor

  # let b = [
  #   [ 7.30f,  1.33,  2.68, -9.62,  0.00],
  #   [ 0.47f,  6.58, -1.71, -0.79,  0.00],
  #   [-6.28f, -3.42,  3.46,  0.41,  0.00]
  # ].toTensor

  # let r = least_square_solver(a, b)

  # echo r

  let A = [
    [0f, 1f],
    [1f, 1f],
    [2f, 1f],
    [3f, 1f]
  ].toTensor

  let y = [-1f, 0.2, 0.9, 2.1].toTensor

  let s = least_square_solver(A, y)

  echo s
