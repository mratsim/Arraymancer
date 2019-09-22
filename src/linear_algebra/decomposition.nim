# Copyright (c) 2018-Present Mamy André-Ratsimbazafy
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

import
  ../tensor/tensor,
  ./helpers/[decomposition_lapack, triangular]

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

  var tau: seq[T]

  geqrf(a, result.Q, tau)
  result.R = triu(result.Q[0..<k, _])

  orgqr(result.Q, tau)
  result.Q = result.Q[_, 0..<k]
