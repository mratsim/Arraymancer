# Copyright (c) 2018 Mamy André-Ratsimbazafy and the Arraymancer contributors
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

import
  ../../tensor/tensor, ../../linear_algebra/linear_algebra


proc pca*[T: SomeReal](x: Tensor[T], nb_components = 2, transform: bool = true): Tensor[T] =
  ## Principal Component Analysis (PCA)
  ## Inputs:
  ##   - A matrix of shape [Nb of observations, Nb of features]
  ##   - The number of components to keep (default 2D for 2D projection)
  ##   - transform: whether or not to output the tensor or components:
  ##     If transform: output the dimensionality reduced tensor
  ##     Else output the first n principal components feature plane
  ##
  ## Returns:
  ##   - A matrix of shape [Nb of observations, Nb of components]

  assert x.rank == 2

  let mean_centered = x .- x.mean(axis=0)

  var cov_matrix = newTensorUninit[T]([x.shape[1], x.shape[1]])
  gemm(1.T / T(x.shape[0]-1), mean_centered.transpose, mean_centered, 0, cov_matrix)

  let (_, eigvecs) = cov_matrix.symeig(true, ^nb_components .. ^1) # Note: eigvals/vecs are returned in ascending order

  # [Nb_obs, Nb_feats] * [Nb_feats, Nb_components], don't forget to reorder component in descending order
  result = mean_centered * eigvecs[_, ^1..0|-1]

  if transform:
    return result
  else:
    return eigvecs

proc pca*[T: SomeReal](x: Tensor[T], principal_axes: Tensor[T]): Tensor[T] =
  ## Principal Component Analysis (PCA) projection
  ## Inputs:
  ##    - A matrix of shape [Nb of observations, Nb of features]
  ##    - A matrix of shape [Nb of observations, Nb of principal axes] to project on
  ##
  ## Returns:
  ##    - A matrix of shape [Nb of observations, Nb of components]
  let mean_centered = x .- x.mean(axis=0)
  result = mean_centered * principal_axes[_, ^1..0|-1]
