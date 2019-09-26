# Copyright (c) 2018-Present Mamy André-Ratsimbazafy
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

import
  nimlapack,
  ./overload,
  ../../private/sequninit,
  ../../tensor/tensor,
  ../../tensor/private/p_init_cpu # TODO: can't call newTensorUninit with optional colMajor, varargs breaks it

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

proc syevr*[T: SomeFloat](a: Tensor[T], eigenvectors: bool,
  low_idx: int, high_idx: int, result: var tuple[eigenval, eigenvec: Tensor[T]]) =
  ## Wrapper for LAPACK syevr routine (Symmetric Recursive Eigenvalue Decomposition)

  assert a.rank == 2, "Input is not a matrix"
  assert a.shape[0] == a.shape[1], "Input should be a symmetric matrix"
  # TODO, support "symmetric matrices" with only the upper or lower part filled.
  # (Obviously, upper in Fortran is lower in C ...)

  let a = a.clone(colMajor) # Lapack overwrites the input. TODO move optimization

  var
    jobz: cstring
    interval: cstring
    n, lda: int32 = a.shape[0].int32
    uplo: cstring = "U"
    vl, vu: T          # unused: min and max eigenvalue returned
    il, iu: int32      # ranking of the lowest and highest eigenvalues returned
    abstol: T = -1     # Use default. otherwise need to call LAPACK routine dlamch('S') or 2*dlamch('S')
    m, ldz = n
    lwork: int32 = -1  # dimension of a workspace array
    work: seq[T]
    wkopt: T
    liwork: int32 = -1 # dimension of a second workspace array
    iwork: seq[cint]
    iwkopt: int32
    info: int32

  if low_idx == 0 and high_idx == a.shape[0] - 1:
    interval = "A"
  else:
    interval = "I"
    il = int32 low_idx + 1 # Fortran starts indexing with 1
    iu = int32 high_idx + 1
    m = iu - il + 1

  # Setting up output
  var
    isuppz: seq[cint] # unused
    isuppz_ptr: ptr int32
    z: ptr T

  result.eigenval = newTensorUninit[T](a.shape[0]) # Even if less eigenval are selected Lapack requires this much workspace

  if eigenvectors:
    jobz = "V"

    # Varargs + optional colMajor argument issue, must resort to low level proc at the moment
    tensorCpu(ldz.int, m.int, result.eigenvec, colMajor)
    result.eigenvec.storage.Fdata = newSeqUninit[T](result.eigenvec.size)

    z = result.eigenvec.get_data_ptr
    if interval == "A": # or (il == 1 and iu == n): -> Already checked before
      isuppz = newSeqUninit[int32](2*m)
      isuppz_ptr = isuppz[0].addr
  else:
    jobz = "N"

  let w = result.eigenval.get_data_ptr

  # Querying workspaces sizes
  syevr(jobz, interval, uplo, n.addr, a.get_data_ptr, lda.addr, vl.addr, vu.addr, il.addr, iu.addr,
        abstol.addr, m.addr, w, z, ldz.addr, isuppz_ptr, wkopt.addr, lwork.addr, iwkopt.addr, liwork.addr, info.addr)

  # Allocating workspace
  lwork = wkopt.int32
  work = newSeqUninit[T](lwork)
  liwork = iwkopt.int32
  iwork = newSeqUninit[int32](liwork)

  # Decompose matrix
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

# QR decomposition
# --------------------------------------------------------------------------------------

# TODO: Batch QR decomposition to lower overhead of intermediates?

overload(geqrf, sgeqrf)
overload(geqrf, dgeqrf)

proc geqrf*[T: SomeFloat](Q: var Tensor[T], tau: var seq[T]) =
  ## Wrapper for LAPACK geqrf routine (GEneral QR Factorization)
  ## Decomposition is done through Householder Reflection
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
  var work = newSeqUninit[T](lwork)

  # Decompose matrix
  geqrf(m.unsafeAddr, n.unsafeAddr, Q.get_data_ptr, m.unsafeAddr, # lda
        tau[0].addr, work[0].addr, lwork.addr, info.addr)
  if unlikely(info < 0):
    raise newException(ValueError, "Illegal parameter in geqrf: " & $(-info))

proc geqrf*[T: SomeFloat](a: Tensor[T], r_v: var Tensor[T], tau: var seq[T]) =
  ## Wrapper for LAPACK geqrf routine (GEneral QR Factorization)
  ## Decomposition is done through Householder Reflection
  ##
  ## Parameters:
  ##   - a: Input - MxN matrix to factorize
  ##   - tau: Output - Scalar factors of elementary Householder Reflectors
  ##   - r_v: Output - MxN matrix
  ##       - R upper-trapezoidal matrix
  ##       - and v vector factors of elementary Householder Reflectors
  ##
  ## Further processing is needed:
  ## - You can extract Q with `orgqr`
  ## - or multiply any matrix by Q without materializing Q with `ormqr`

  assert a.rank == 2, "Input is not a matrix"

  # Lapack overwrites the input
  #   - contains R above the diagonal
  #     - min(M,N)-by-N upper trapezoidal matrix
  #     - if M > N, R is upper triangular
  #   - contains V, a vector that needs to be multiplied
  #     to TAU to reconstruct Q

  # Outputs
  r_v = a.clone(colMajor)
  tau = newSeqUninit[T](min(r_v.shape[0], r_v.shape[1]))

  geqrf(r_v, tau)

overload(orgqr, sorgqr)
overload(orgqr, dorgqr)

proc orgqr*[T: SomeFloat](rv_q: var Tensor[T], tau: openarray[T]) =
  ## Wrapper for LAPACK orgqr routine
  ## Generates the orthonormal Q matrix from
  ## elementary Householder reflectors
  ##
  ## Inputs **must** come from a previous geqrf
  ##   - rv_q: contains r_v on input. A column-major vector factors of elementary reflectors
  ##   - tau: Scalar factors of elementary reflectors
  ##
  ## Outputs
  ##   - rv_q: overwritten by Q
  ##
  ## Note that while rv_q is MxN on input
  ## on output the shape is M x min(M,N)
  ##
  ## ⚠️: Output must be sliced by [M, min(M,N)]
  ##    if M>N as the rest contains garbage
  ##
  ## Spec: https://www.nag.co.uk/numeric/fl/nagdoc_fl24/pdf/f08/f08aff.pdf
  ## API: http://www.netlib.org/lapack/explore-html/da/dba/group__double_o_t_h_e_rcomputational_ga14b45f7374dc8654073aa06879c1c459.html
  assert rv_q.rank == 2
  assert rv_q.is_F_contiguous()

  let
    m = rv_q.shape[0].int32                     # Order of the orthonormal matrix Q
    n = int32 min(rv_q.shape[0], rv_q.shape[1]) # Number of columns of Q
    k = n                                       # The number of elementary reflectors whose product defines the matrix Q
  var
    # LAPACK stores optimal scratchspace size in the first element of a float array ...
    work_size: T
    lwork = -1'i32 # size query
    info: int32

  assert k == tau.len

  # Querying workspace size
  orgqr(m.unsafeAddr, n.unsafeAddr, k.unsafeAddr, rv_q.get_data_ptr, m.unsafeAddr, # lda
        tau[0].unsafeAddr, work_size.addr, lwork.addr, info.addr)

  # Allocating workspace
  lwork = work_size.int32
  var work = newSeqUninit[T](lwork)

  # Extract Q from Householder reflectors
  orgqr(m.unsafeAddr, n.unsafeAddr, k.unsafeAddr, rv_q.get_data_ptr, m.unsafeAddr, # lda
        tau[0].unsafeAddr, work[0].addr, lwork.addr, info.addr)
  if unlikely(info < 0):
    raise newException(ValueError, "Illegal parameter in geqrf: " & $(-info))

# Singular Value decomposition
# --------------------------------------------------------------------------------------

overload(gesdd, sgesdd)
overload(gesdd, dgesdd)

proc gesdd*[T: SomeFloat](a: Tensor[T], U, S, Vh: var Tensor[T]) =
  ## Wrapper for LAPACK gesdd routine
  ## (GEneral Singular value Decomposition by Divide & conquer)
  ##
  ## Parameters:
  ##   - a: Input - MxN matrix to factorize
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

  # - https://software.intel.com/en-us/node/469238
  # - http://www.netlib.org/lapack/explore-html/d4/dca/group__real_g_esing_gac2cd4f1079370ac908186d77efcd5ea8.html
  # - https://software.intel.com/sites/products/documentation/doclib/mkl_sa/11/mkl_lapack_examples/sgesdd_ex.c.htm

  assert a.rank == 2
  let a = a.clone(colMajor) # Lapack destroys the input. TODO newruntime sink if possible

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

  # newTensorUninit: Varargs + optional colMajor argument issue, must resort to low level proc at the moment
  # U
  tensorCpu([ldu.int, ucol.int], U, colMajor)
  U.storage.Fdata = newSeqUninit[T](U.size)
  # S
  S = newTensorUninit[T](k.int)
  # V.H
  tensorCpu([ldvt.int, n.int], Vh, colMajor)
  Vh.storage.Fdata = newSeqUninit[T](Vh.size)

  # Querying workspace size
  gesdd(jobz, m.unsafeAddr, n.unsafeAddr,
        a.get_data_ptr, m.unsafeAddr, # lda
        S.get_data_ptr,
        U.get_data_ptr, ldu.unsafeAddr,
        Vh.get_data_ptr, ldvt.unsafeAddr,
        work_size.addr, lwork.addr, iwork[0].addr,
        info.addr
        )

  # Allocating workspace
  lwork = work_size.int32
  var work = newSeqUninit[T](lwork)

  # Decompose matrix
  gesdd(jobz, m.unsafeAddr, n.unsafeAddr,
        a.get_data_ptr, m.unsafeAddr, # lda
        S.get_data_ptr,
        U.get_data_ptr, ldu.unsafeAddr,
        Vh.get_data_ptr, ldvt.unsafeAddr,
        work[0].addr, lwork.addr, iwork[0].addr,
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

proc getrf*[T: SomeFloat](a: Tensor[T], lu: var Tensor[T], pivot_indices: var seq[int32]) =
  ## Wrapper for LAPACK getrf routine
  ## (GEneral ??? Pivoted LU Factorization)
  ##
  ## Input:
  ##   - a: MxN matrix to factorize
  ##   - lu: MxN matrix:
  ##       - Upper triangle including diagonal contains U
  ##       - Lower triangle contains L, L has an unit diagonal that needs to be reconstructed
  ##   - pivot_indices: indices of the permutation matrix P
  ##                    sparse representation:
  ##                    | 1 0 0 0 |
  ##                    | 0 0 1 0 |
  ##                    | 0 0 0 1 |
  ##                    | 0 1 0 0 |
  ##                    would be @[0, 2, 3, 1]
  ##
  ## The inputs and outputs solve the equation
  ## A = P L U

  assert a.rank == 2
  let k = min(a.shape[0], a.shape[1]).int

  # Outputs
  lu = a.clone(colMajor) # Lapack destroys the input. TODO newruntime sink if possible
  pivot_indices = newSeqUninit[int32](k)

  getrf(lu, pivot_indices)

# Sanity checks
# --------------------------------------------------------------------------------------

when isMainModule:
  import ../../ml/metrics/common_error_functions

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

    var tau: seq[float64]
    var r_v: Tensor[float64]
    geqrf(a, r_v, tau)

    doAssert mean_absolute_error(r_v, expected_rv) < 1e-6
    doAssert mean_absolute_error(tau.toTensor, expected_tau) < 1e-15
