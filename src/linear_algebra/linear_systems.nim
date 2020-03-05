# Copyright (c) 2019-Present the Arraymancer contributors
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

import
  strformat,
  ../tensor/tensor,
  ../private/sequninit,
  ./helpers/solve_lapack

type MatrixKind* = enum
    mkGeneral,
    mkGenBand,
    mkGenTriDiag,
    mkSymmetric,
    mkPosDef,
    mkPosDefBand,
    mkPosDefTriDiag

proc solve*[T: SomeFloat](a, b: Tensor[T], kind: MatrixKind = mkGeneral): Tensor[T] =
  ## Compute the solution ``X`` to the system of linear equations ``AX = B``.
  ##
  ## Multiple right-hand sides can be solved simultaneously.
  ##
  ## Input:
  ##   - ``a``, a MxM matrix
  ##   - ``b``, a vector of length M, or a MxN matrix. In the latter case, each
  ##     column is interpreted as a separate RHS to solve for.
  ##
  ## Output:
  ##   - Tensor with same shape as ``b``
  assert a.rank == 2
  assert b.rank <= 2

  let k = min(a.shape[0], a.shape[1])

  # mutated by lapack wrappers
  var
    pivot_indices = newSeqUninit[int32](k)
    lu = a.clone(colMajor)
  result = b.clone(colMajor)

  # Automatically handle single or multiple RHS
  if result.rank == 1:
    result = result.reshape(result.shape[0], 1)
  assert result.rank == 2

  case kind:
    of mkGeneral:
      gesv(lu, result, pivot_indices)
    else:
      # TODO: add support for other matrices supported by LAPACK
      raise newException(ValueError,
                         fmt"solve not implemented for matrix kind {kind}")

  # If single RHS, return vector
  if b.rank == 1:
    result = result.reshape(result.shape[0])
  assert result.shape == b.shape
