# Copyright (c) 2018-Present Mamy André-Ratsimbazafy
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

import
  ../tensor/tensor,
  ../private/sequninit,
  ./helpers/[decomposition_lapack, triangular, auxiliary_lapack]

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

  syevr(a, eigenvectors, 0, a.shape[0] - 1, result)

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

  syevr(a, eigenvectors, a ^^ slice.a, a ^^ slice.b, result)

proc qr*[T: SomeFloat](a: Tensor[T]): tuple[Q, R: Tensor[T]] =
  ## Compute the QR decomposition of an input matrix ``a``
  ## Decomposition is done through the Householder method
  ##
  ## Input:
  ##   - ``a``, matrix of shape [M, N]
  ##
  ## We note K = min(M, N)
  ##
  ## Returns:
  ##   - Q orthonormal matrix of shape [M, K]
  ##   - R upper-triangular matrix of shape [K, N]

  let k = min(a.shape[0], a.shape[1])

  var scratchspace: seq[T]
  var tau = newSeqUninit[T](k)

  result.Q = a.clone(colMajor)

  geqrf(result.Q, tau, scratchspace)
  result.R = triu(result.Q[0..<k, _])

  orgqr(result.Q, tau, scratchspace)
  result.Q = result.Q[_, 0..<k]

proc svd*[T: SomeFloat](a: Tensor[T]): tuple[U, S, Vh: Tensor[T]] =
  ## Compute the Singular Value Decomposition of an input matrix ``a``
  ## Decomposition is done through recursive divide & conquer.
  ##
  ## Input:
  ##   - ``a``, matrix of shape [M, N]
  ##
  ## Returns:
  ##   with K = min(M, N)
  ##   - ``U``: Unitary matrix of shape [M, K] with left singular vectors as columns
  ##   - ``S``: Singular values diagonal of length K in decreasing order
  ##   - ``Vh``: Unitary matrix of shape [K, N] with right singular vectors as rows
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
  ## Compared to Numpy svd procedure, we default to "full_matrices = false".
  ##
  ## Exception:
  ##   - This can throw if the algorithm did not converge.

  # Numpy default to full_matrices is
  # - confusing for docs
  # - hurts reconstruction
  # - often not used (for example for PCA/randomized PCA)
  # - memory inefficient
  # thread: https://mail.python.org/pipermail/numpy-discussion/2011-January/054685.html
  var scratchspace: seq[T]
  gesdd(a, result.U, result.S, result.Vh, scratchspace)

proc lu_permuted*[T: SomeFloat](a: Tensor[T]): tuple[PL, U: Tensor[T]] =
  ## Compute the pivoted LU decomposition of an input matrix ``a``.
  ##
  ## The decomposition solves the equation:
  ## A = P L U
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
  ##   - U, upper-triangular matrix of shape [K, N]
  assert a.rank == 2

  let k = min(a.shape[0], a.shape[1])
  var pivot_indices = newSeqUninit[int32](k)
  var lu = a.clone(colMajor)

  getrf(lu, pivot_indices)

  result.U = triu(lu[0..<k, _])
  result.PL = tril_unit_diag(lu[_, 0..<k])
  laswp(result.PL, pivot_indices, pivot_from = -1)
