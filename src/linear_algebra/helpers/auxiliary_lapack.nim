# Copyright (c) 2018-Present Mamy André-Ratsimbazafy
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

import
  nimlapack,
  ./overload,
  ../../tensor/tensor

# Auxiliary functions from Lapack
# ----------------------------------

overload(laswp, slaswp)
overload(laswp, dlaswp)

proc laswp*(a: var Tensor, pivot_indices: openarray[int32], pivot_from: static int32) =
  ## Apply A = P * A
  ## where P is a permutation matrix, represented pivot_indices of rows.
  ##
  ## A is a matrix of shape MxN.
  ## A is permuted in-place

  # http://www.netlib.org/lapack/explore-html/d1/db5/group__real_o_t_h_e_rauxiliary_gacb14404955e1b301d7877892a3c83f3d.html#gacb14404955e1b301d7877892a3c83f3d

  assert a.rank == 2
  assert a.is_F_contiguous()

  let
    k = min(a.shape[0], a.shape[1]).int32
    n = k
    lda = a.shape[0].int32  # Assumes colMajor
    k1 = 1'i32              # First element of pivot_indices that will be used for row interchange
    k2 = k                  # Last element of pivot_indices that will be used for row interchanged
    incx = pivot_from.int32 # 1: multiply by Permutation on the right
                            # -1: multiply by permutation on the left

  assert k == pivot_indices.len

  laswp(n.unsafeAddr, a.get_data_ptr, lda.unsafeAddr, k1.unsafeAddr, k2.unsafeAddr,
        pivot_indices[0].unsafeAddr, incx.unsafeAddr)


overload(orgqr, sorgqr)
overload(orgqr, dorgqr)

proc orgqr*[T: SomeFloat](rv_q: var Tensor[T], tau: openarray[T], scratchspace: var seq[T]) =
  ## Wrapper for LAPACK orgqr routine
  ## Generates the orthonormal Q matrix from
  ## elementary Householder reflectors
  ##
  ## Inputs **must** come from a previous geqrf
  ##   - rv_q: contains r_v (reflector vector) on input.
  ##          A column-major vector factors of elementary reflectors
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
        tau[0].unsafeAddr, work_size.addr,
        lwork.addr, info.addr)

  # Allocating workspace
  lwork = work_size.int32
  scratchspace.setLen(lwork)

  # Extract Q from Householder reflectors
  orgqr(m.unsafeAddr, n.unsafeAddr, k.unsafeAddr, rv_q.get_data_ptr, m.unsafeAddr, # lda
        tau[0].unsafeAddr, scratchspace[0].addr,
        lwork.addr, info.addr)
  if unlikely(info < 0):
    raise newException(ValueError, "Illegal parameter in geqrf: " & $(-info))

overload(ormqr, sormqr)
overload(ormqr, dormqr)

proc ormqr*[T: SomeFloat](C: var Tensor[T], Q: Tensor[T], tau: openarray[T], side, trans: static char, scratchspace: var seq[T]) =
  ## Wrapper for LAPACK ormqr routine
  ## Multiply the orthonormal Q matrix from geqrf
  ## with another matrix C without materializing Q
  ##
  ## C is a matrix of shae [M, N] and will be overwritten by
  ##
  ##                SIDE = 'L'     SIDE = 'R'
  ## TRANS = 'N':      Q * C          C * Q
  ## TRANS = 'T':      Q**T * C       C * Q**T

  assert C.rank == 2
  assert C.is_F_contiguous()
  assert Q.rank == 2
  assert Q.is_F_contiguous()
  static:
    assert trans in {'N', 'T'}

  let
    m = C.shape[0].int32 # number of rows of C
    n = C.shape[1].int32 # number of columns of C
    k = Q.shape[1].int32 # The number of elementary reflectors whose product defines the matrix Q

    lda = Q.shape[0].int32
    sside = cstring($side)
    strans = cstring($trans)

  assert k == tau.len
  when side == 'L':
    assert lda == m
  elif side == 'R':
    assert lda == n
  else:
    {.error: "Only L(eft) and R(ight) are valid inputs for side".}

  var
    work_size: T
    lwork = -1'i32 # size query
    info: int32

  # Querying workspace size
  ormqr(sside, strans, m.unsafeAddr, n.unsafeAddr, k.unsafeAddr,
        Q.get_data_ptr, lda.unsafeAddr, tau[0].unsafeAddr,
        C.get_data_ptr, m.unsafeAddr, # ldc
        work_size.addr,
        lwork.addr, info.addr
      )

  # Allocating workspace
  lwork = work_size.int32
  scratchspace.setLen(lwork)

  # Matrix multiplication
  ormqr(sside, strans, m.unsafeAddr, n.unsafeAddr, k.unsafeAddr,
        Q.get_data_ptr, lda.unsafeAddr, tau[0].unsafeAddr,
        C.get_data_ptr, m.unsafeAddr, # ldc
        scratchspace[0].addr,
        lwork.addr, info.addr
      )
  if unlikely(info < 0):
    raise newException(ValueError, "Illegal parameter in ormqr: " & $(-info))


# Sanity checks
# -----------------------------------------------

when isMainModule:
  import ./decomposition_lapack
  import ../../ml/metrics/common_error_functions
  import ../../private/sequninit

  let a = [[12.0, -51.0, 4.0],
          [ 6.0, 167.0, -68.0],
          [-4.0,  24.0, -41.0]].toTensor()

  # From numpy
  let np_q = [[-0.85714286,  0.39428571,  0.33142857],
              [-0.42857143, -0.90285714, -0.03428571],
              [ 0.28571429, -0.17142857,  0.94285714]].toTensor()

  let k = min(a.shape[0], a.shape[1])
  var Q_reflectors: Tensor[float64]
  var tau = newSeqUninit[float64](k)
  var scratchspace: seq[float64]

  # QR decomposition
  Q_reflectors = a.clone(colMajor)
  geqrf(Q_reflectors, tau, scratchspace)

  # Materialize Q
  var Q = Q_reflectors.clone(colMajor)
  orgqr(Q, tau, scratchspace)
  doAssert Q.mean_absolute_error(np_q) < 1e-8

  # Check multiplication
  let Msrc = [[1.0, 2, 3],
           [4.0, 5, 6],
           [7.0, 8, 9]].toTensor()

  block: # M*Q
    var M = Msrc.clone(colMajor)
    ormqr(M, Q_reflectors, tau, side = 'R', trans = 'N', scratchspace)
    doAssert M.mean_absolute_error(Msrc * Q) < 1e-8

  block: # Q*M
    var M = Msrc.clone(colMajor)
    ormqr(M, Q_reflectors, tau, side = 'L', trans = 'N', scratchspace)
    doAssert M.mean_absolute_error(Q * Msrc) < 1e-8

  block: # M*Q.T
    var M = Msrc.clone(colMajor)
    ormqr(M, Q_reflectors, tau, side = 'R', trans = 'T', scratchspace)
    doAssert M.mean_absolute_error(Msrc * Q.transpose()) < 1e-8

  block: # Q.T * M
    var M = Msrc.clone(colMajor)
    ormqr(M, Q_reflectors, tau, side = 'L', trans = 'T', scratchspace)
    doAssert M.mean_absolute_error(Q.transpose() * Msrc) < 1e-8
