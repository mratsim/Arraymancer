# Copyright (c) 2018 Mamy André-Ratsimbazafy and the Arraymancer contributors
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

import ../tensor

proc covariance_matrix*[T: SomeFloat](x, y: Tensor[T]): Tensor[T] =

  ## Input:
  ##   - 2 tensors of shape [Nb observations, features]
  ##     Note: contrary to Numpy default each row is an observations while
  ##           echo column represent a feature/variable observed.
  ## Returns:
  ##   - The unbiased covariance (normalized by the number of observations - 1)
  ##     in the shape [features, features]

  # The covariance is cov(X, Y) = 1/(n-1) * ∑i->n (Xi - X.mean)(Yi - Y.mean)
  # The covariance matrix is generalizing this for all Xi and Yi of both matrices:
  # https://www.itl.nist.gov/div898/handbook/pmc/section5/pmc541.htm
  # And
  # http://www.cs.otago.ac.nz/cosc453/student_tutorials/principal_components.pdf

  # TODO proper checks
  assert x.rank == 2
  assert x.shape == y.shape

  let deviation_X = (x -. x.mean(axis=0)).transpose # shape [features, batch_size]
  let deviation_Y = y -. y.mean(axis=0)             # shape [batch_size, features]

  result = newTensorUninit[T]([x.shape[1], x.shape[1]])
  gemm(1.T / T(x.shape[0]-1), deviation_X, deviation_Y, 0, result)
