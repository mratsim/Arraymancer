# Copyright (c) 2018 the Arraymancer contributors
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

import std/complex
import
  ../tensor,
  ./complex

proc pinv*[T: SomeFloat](A: Tensor[T], rcond = 1e-15): Tensor[T] =
  ## Compute the (Moore-Penrose) pseudo-inverse of a matrix.
  ##
  ## Calculate the generalized inverse of a matrix using its
  ## singular-value decomposition (SVD) and including all
  ## *large* singular values.
  ##
  ## Input:
  ##   - A: the rank-2 tensor to invert
  ##   - rcond: Cutoff ratio for small singular values.
  ##            Singular values less than or equal to
  ##            `rcond * largest_singular_value` are set to zero.
  var (U, S, Vt) = A.svd()
  let epsilon = S.max() * rcond
  let S_size = S.shape[0]
  var S_inv = zeros[T]([S_size, S_size])
  const unit = T(1.0)
  for n in 0..<S.shape[0]:
    S_inv[n, n] = if abs(S[n]) > epsilon: unit / S[n] else: 0.0
  result = Vt.transpose * S_inv * U.transpose

proc pinv*[T: Complex32 | Complex64](A: Tensor[T], rcond = 1e-15): Tensor[T] =
  ## Compute the (Moore-Penrose) pseudo-inverse of a matrix.
  ##
  ## Calculate the generalized inverse of a matrix using its
  ## singular-value decomposition (SVD) and including all
  ## *large* singular values.
  ##
  ## Input:
  ##   - A: the rank-2 tensor to invert
  ##   - rcond: Cutoff ratio for small singular values.
  ##            Singular values less than or equal to
  ##            `rcond * largest_singular_value` are set to zero.
  var (U, S, Vt) = A.svd()
  let epsilon = S.max() * rcond
  let S_size = S.shape[0]
  var S_inv = zeros[T]([S_size, S_size])
  const unit = complex(1.0)
  for n in 0..<S.shape[0]:
    S_inv[n, n] = if abs(S[n]) > epsilon: unit / S[n] else: complex(0.0)
  result = Vt.conjugate.transpose * S_inv * U.conjugate.transpose
