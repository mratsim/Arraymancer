# Copyright (c) 2018-Present Mamy André-Ratsimbazafy
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

import
  ./decomposition_lapack

# LU Factorization - private implementations
# -------------------------------------------

proc lu_permuted_inplace*(a: var Tensor) =
  ## Compute the pivoted LU decomposition of an input matrix ``a``.
  ## This overwrites the input matrix with the pivoted lower triangular matrix
  ##
  ## The decomposition solves the equation:
  ## A = P L U
  ##
  ## And the procedure returns PL
  ##
  ## where:
  ##   - P is a permutation matrix
  ##   - L is a lower-triangular matrix with unit diagonal
  ##   - U is an upper-triangular matrix
  ##
  ## Input:
  ##   - ``a``, a MxN matrix
  ##
  ## Output:
  ##   with K = min(M, N)
  ##   - PL, the product of P and L, of shape [M, K]

  assert a.rank == 2
  assert a.is_F_contiguous

  let k = min(a.shape[0], a.shape[1])
  var pivot_indices = newSeqUninit[int32](k)

  getrf(a, pivot_indices)

  a = a[_, 0..<k]
  tril_unit_diag_mut(a)
  laswp(a, pivot_indices, pivot_from = -1)
