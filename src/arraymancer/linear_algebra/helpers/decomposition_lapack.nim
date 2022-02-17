# Copyright (c) 2018-Present Mamy André-Ratsimbazafy
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

import
  nimlapack,
  ./overload,
  ./init_colmajor,
  ../../private/sequninit,
  ../../tensor

# Wrappers for Fortran LAPACK
# We don't use the C interface LAPACKE that
# automatically manages the work arrays:
#   - it's yet another dependency
#   - it cannot deal with strided arrays

# TODO: we have to use seq[cint] instead of seq[int32]
#       in several places due to nimlapack using cint
#       and bad C++ codegen of seq[int32]
#       - https://github.com/nim-lang/Nim/issues/5905
#       - https://github.com/nim-lang/Nim/issues/7308
#       - https://github.com/nim-lang/Nim/issues/11797

# SVD and Eigenvalues/eigenvectors decomposition
# --------------------------------------------------------------------------------------

overload(syevr, ssyevr)
overload(syevr, dsyevr)

proc syevr*[T: SomeFloat](a: var Tensor[T], uplo: static char, return_eigenvectors: static bool,
  low_idx: int, high_idx: int, eigenval, eigenvec: var Tensor[T], scratchspace: var seq[T]) =
  ## Wrapper for LAPACK syevr routine (Symmetric Recursive Eigenvalue Decomposition)
  ##
  ## eigenvalues are returned in ascending order
  ## (from lower to upper)
  ##
  ## if uplo = 'L', the lower part of A is used
  ## it is destroyed on exit (and upper part is untouched)
  ## vice-versa if uplo = 'U' for the upper part of A

  assert a.rank == 2, "Input is not a matrix"
  assert a.shape[0] == a.shape[1], "Input should be a symmetric matrix"

  static:
    assert uplo in {'L', 'U'}

  var uplo_layout: cstring
  if a.is_F_contiguous:
    uplo_layout = static(cstring($uplo))
  elif a.is_C_contiguous:
    # For row major layouts we take the opposite triangle from specified
    when uplo == 'L':
      uplo_layout = static(cstring"U")
    else:
      uplo_layout = static(cstring"L")
  else:
    raise newException(ValueError, "Input of syevr is not contiguous")

  let n = a.shape[0].int32
  var m: int32

  var
    jobz: cstring
    interval: cstring
    vl, vu: T          # unused: min and max eigenvalue returned
    il, iu: int32      # ranking of the lowest and highest eigenvalues returned
    abstol: T = -1     # Use default. otherwise need to call LAPACK routine dlamch('S') or 2*dlamch('S')
    lwork: int32 = -1  # dimension of a workspace array
    work_size: T
    liwork: int32 = -1 # dimension of a second workspace array
    iwork: seq[cint]
    iwork_size: int32
    info: int32

  if low_idx == 0 and high_idx == a.shape[0] - 1:
    interval = "A"
    m = n
  else:
    interval = "I"
    il = int32 low_idx + 1 # Fortran starts indexing with 1
    iu = int32 high_idx + 1
    m = iu - il + 1

  # Setting up output
  var
    isuppz: seq[cint] # unused
    isuppz_ptr: ptr int32

  eigenval = newTensorUninit[T](a.shape[0]) # Even if less eigenval are selected Lapack requires this much workspace

  if return_eigenvectors:
    jobz = "V"
    eigenvec.newMatrixUninitColMajor(n, m) # ldz, m
    if interval == "A": # or (il == 1 and iu == n): -> Already checked before
      isuppz = newSeqUninit[int32](2*m)
      isuppz_ptr = isuppz[0].addr
  else:
    jobz = "N"

  # Querying workspaces sizes
  syevr(jobz, interval, uplo_layout,
        n.unsafeAddr,
        a.get_data_ptr, n.unsafeAddr, # lda
        vl.addr, vu.addr, il.addr, iu.addr,
        abstol.addr, m.addr,
        eigenval.get_data_ptr,
        eigenvec.get_data_ptr, n.unsafeAddr, # ldz
        isuppz_ptr,
        work_size.addr, lwork.addr,
        iwork_size.addr, liwork.addr, info.addr)

  # Allocating workspace
  lwork = work_size.int32
  scratchspace.setLen(lwork)
  liwork = iwork_size.int32
  iwork = newSeqUninit[int32](liwork)

  # Decompose matrix
  syevr(jobz, interval, uplo_layout,
        n.unsafeAddr,
        a.get_data_ptr, n.unsafeAddr,
        vl.addr, vu.addr, il.addr, iu.addr,
        abstol.addr, m.addr,
        eigenval.get_data_ptr,
        eigenvec.get_data_ptr, n.unsafeAddr,
        isuppz_ptr,
        scratchspace[0].addr, lwork.addr,
        iwork[0].addr, liwork.addr, info.addr)

  # Keep only the selected eigenvals
  eigenval = eigenval[0 ..< m.int]

  when compileOption("boundChecks"):
    if return_eigenvectors:
      assert m.int == eigenvec.shape[1]

  if unlikely(info > 0):
    # TODO, this should not be an exception, not converging is something that can happen and should
    # not interrupt the program. Alternative. Fill the result with Inf?
    raise newException(ValueError, "the algorithm for computing eigenvalues failed to converge")

  if unlikely(info < 0):
    raise newException(ValueError, "Illegal parameter in symeig: " & $(-info))

# QR decomposition
# --------------------------------------------------------------------------------------

# TODO: Batch QR decomposition to lower overhead of intermediates?

overload(geqrf, sgeqrf)
overload(geqrf, dgeqrf)

proc geqrf*[T: SomeFloat](Q: var Tensor[T], tau: var seq[T], scratchspace: var seq[T]) =
  ## Wrapper for LAPACK geqrf routine (GEneral QR Factorization)
  ## Decomposition is done through Householder Reflection
  ## and without pivoting
  ##
  ## In-place version, this will overwrite Q and tau
  assert Q.rank == 2, "Input is not a matrix."
  assert Q.is_F_contiguous, "Input must be column-major."
  assert tau.len == min(Q.shape[0], Q.shape[1]), "tau should have the size of min(Q.shape[0], Q.shape[1])"

  # Temporaries
  let
    m = Q.shape[0].int32 # colMajor for Fortran
    n = Q.shape[1].int32
  var
    # LAPACK stores optimal scratchspace size in the first element of a float array ...
    work_size: T
    lwork = -1'i32 # size query
    info: int32

  # Querying workspace size
  geqrf(m.unsafeAddr, n.unsafeAddr, Q.get_data_ptr, m.unsafeAddr, # lda
        tau[0].addr, work_size.addr, lwork.addr, info.addr)

  # Allocating workspace
  lwork = work_size.int32
  scratchspace.setLen(lwork)

  # Decompose matrix
  geqrf(m.unsafeAddr, n.unsafeAddr, Q.get_data_ptr, m.unsafeAddr, # lda
        tau[0].addr, scratchspace[0].addr, lwork.addr, info.addr)
  if unlikely(info < 0):
    raise newException(ValueError, "Illegal parameter in geqrf: " & $(-info))

# Singular Value decomposition
# --------------------------------------------------------------------------------------

overload(gesdd, sgesdd)
overload(gesdd, dgesdd)

proc gesdd*[T: SomeFloat](a: var Tensor[T], U, S, Vh: var Tensor[T], scratchspace: var seq[T]) =
  ## Wrapper for LAPACK gesdd routine
  ## (GEneral Singular value Decomposition by Divide & conquer)
  ##
  ## Parameters:
  ##   - a: Input - MxN matrix to factorize, in column major format
  ##   - U: Output - Unitary matrix containing the left singular vectors as columns
  ##   - S: Output - Singular values sorted in decreasing order
  ##   - Vh: Output - Unitary matrix containing the right singular vectors as rows
  ##
  ## SVD solves the equation:
  ## A = U S V.h
  ##
  ## - with S being a diagonal matrix of singular values
  ## - with V being the right singular vectors and
  ##   V.h being the hermitian conjugate of V
  ##   for real matrices, this is equivalent to V.t (transpose)
  ##
  ## ⚠️: Input must not contain NaN
  ##
  ## Performance note:
  ## - Lapack, especially with the OpenBLAS backend is much more optimized
  ##   for input [M, N] where M > N versus N < M (2x - 3x speed difference)
  ##   Transpose accordingly.
  ##   Matrices must be column major.

  # - https://software.intel.com/en-us/node/469238
  # - http://www.netlib.org/lapack/explore-html/d4/dca/group__real_g_esing_gac2cd4f1079370ac908186d77efcd5ea8.html
  # - https://software.intel.com/sites/products/documentation/doclib/mkl_sa/11/mkl_lapack_examples/sgesdd_ex.c.htm

  assert a.rank == 2
  assert a.is_F_contiguous()

  # Temporaries
  let
    m = a.shape[0].int32 # colMajor in Fortran
    n = a.shape[1].int32
    # Returns reduced columns in U (shape Mxk)
    # and reduced rows in Vh (shape kxN)
    # Numpy default to full_matrices is
    # - confusing for docs
    # - hurts reconstruction
    # - often not used (for example for PCA/randomized PCA)
    # - memory inefficient
    # thread: https://mail.python.org/pipermail/numpy-discussion/2011-January/054685.html
    jobz = cstring"S"
    k = min(m, n)
    ldu = m # depends on jobz
    ucol = k # depends on jobz
    ldvt = k # depends on jobz
  var
    # LAPACK stores optimal scratchspace size in the first element of a float array ...
    work_size: T
    lwork = -1'i32 # size query
    info: int32
    iwork = newSeqUninit[cint](8 * k)

  U.newMatrixUninitColMajor(ldu, ucol)
  S = newTensorUninit[T](k.int)
  Vh.newMatrixUninitColMajor(ldvt, n)

  # Querying workspace size
  gesdd(jobz, m.unsafeAddr, n.unsafeAddr,
        a.get_data_ptr, m.unsafeAddr, # lda
        S.get_data_ptr,
        U.get_data_ptr, ldu.unsafeAddr,
        Vh.get_data_ptr, ldvt.unsafeAddr,
        work_size.addr,
        lwork.addr, iwork[0].addr,
        info.addr
       )

  # Allocating workspace
  lwork = work_size.int32
  scratchspace.setLen(lwork)

  # Decompose matrix
  gesdd(jobz, m.unsafeAddr, n.unsafeAddr,
        a.get_data_ptr, m.unsafeAddr, # lda
        S.get_data_ptr,
        U.get_data_ptr, ldu.unsafeAddr,
        Vh.get_data_ptr, ldvt.unsafeAddr,
        scratchspace[0].addr,
        lwork.addr, iwork[0].addr,
        info.addr
       )

  if info > 0:
    # TODO, this should not be an exception, not converging is something that can happen and should
    # not interrupt the program. Alternative. Fill the result with Inf?
    # Though scipy / sklearn seems to use LinAlgError exceptions
    raise newException(ValueError, "the algorithm for computing the SVD failed to converge")
  if info < 0:
    raise newException(ValueError, "Illegal parameter in singular value decomposition gesdd: " & $(-info))

# Pivoted LU Factorization
# --------------------------------------------------------------------------------------
overload(getrf, sgetrf)
overload(getrf, dgetrf)

proc getrf*[T: SomeFloat](lu: var Tensor[T], pivot_indices: var seq[int32]) =
  ## Wrapper for LAPACK getrf routine
  ## (GEneral ??? Pivoted LU Factorization)
  ##
  ## In-place version, this will overwrite LU and tau
  assert lu.rank == 2, "Input is not a matrix"
  assert lu.is_F_contiguous, "Input must be column-major"
  assert pivot_indices.len == min(lu.shape[0], lu.shape[1]), "pivot_indices should have the size of min(Q.shape[0], Q.shape[1])"

  # Temporaries
  let
    m = lu.shape[0].int32 # colMajor in Fortran
    n = lu.shape[1].int32
  var info: int32

  # Decompose matrix
  getrf(m.unsafeAddr, n.unsafeAddr, lu.get_data_ptr, m.unsafeAddr, # lda
        pivot_indices[0].addr, info.addr)
  if info < 0:
    raise newException(ValueError, "Illegal parameter in lu factorization getrf: " & $(-info))
  if info > 0:
    # TODO: warning framework
    let cinfo = $(info - 1) # Fortran arrays are indexed by 1
    echo "Warning: in LU factorization, diagonal U[" & cinfo & "," & cinfo & "] is zero. Matrix is singular/non-invertible.\n" &
      "Division-by-zero will occur if used to solve a system of equations."

# Sanity checks
# --------------------------------------------------------------------------------------

when isMainModule:
  import ../../ml/metrics/common_error_functions

  block:
    let a = [[ 1.0, 2.0],
             [ 2.0, 1.0]].toTensor()

    let expected_eigvals = [-1.0, 3.0].toTensor()
    let expected_eigvecs = [[-0.7071067811865475, 0.7071067811865475],
                            [0.7071067811865475, 0.7071067811865475]].toTensor()

    var scratchspace: seq[float64]
    var A = a.clone(colMajor)
    var eigenval: Tensor[float64]
    var eigenvec: Tensor[float64]

    syevr(A, 'U', true, 0, a.shape[0] - 1, eigenval, eigenvec, scratchspace)

    doAssert expected_eigvals.mean_absolute_error(eigenval) < 1e-15
    doAssert expected_eigvecs.mean_absolute_error(eigenvec) < 1e-15

  block: # QR decompositions
    # Adapted from: https://www.ibm.com/support/knowledgecenter/en/SSFHY8_6.2/reference/am5gr_hdgeqrf.html

    # --- Input ---------------------
    #         |   .000000  2.000000 |
    #         |  2.000000 -1.000000 |
    # A    =  |  2.000000 -1.000000 |
    #         |   .000000  1.500000 |
    #         |  2.000000 -1.000000 |
    #         |  2.000000 -1.000000 |
    # --- Output ---------------------
    #         | -4.000000  2.000000 |
    #         |   .500000  2.500000 |
    # A    =  |   .500000   .285714 |
    #         |   .000000  -.428571 |
    #         |   .500000   .285714 |
    #         |   .500000   .285714 |
    #
    # TAU  =  |  1.000000  1.400000 |

    let a = [[ 0.0,  2.0],
            [ 2.0, -1.0],
            [ 2.0, -1.0],
            [ 0.0,  1.5],
            [ 2.0, -1.0],
            [ 2.0, -1.0]].toTensor()

    let expected_rv = [[ -4.0, 2.0],
                      [  0.5, 2.5],
                      [  0.5, 0.285714],
                      [  0.0,-0.428571],
                      [  0.5, 0.285714],
                      [  0.5, 0.285714]].toTensor()
    let expected_tau = [1.0, 1.4].toTensor()

    let k = min(a.shape[0], a.shape[1])
    var tau = newSeqUninit[float64](k)
    var r_v = a.clone(colMajor)
    var scratchspace: seq[float64]
    geqrf(r_v, tau, scratchspace)

    doAssert r_v.mean_absolute_error(expected_rv) < 1e-6
    doAssert tau.toTensor.mean_absolute_error(expected_tau) < 1e-15
