# Copyright (c) 2018 Mamy André-Ratsimbazafy and the Arraymancer contributors
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

import
  ../../tensor/tensor, ../../linear_algebra/linear_algebra


proc pca*[T: SomeReal](x: Tensor[T], nb_components = 2): tuple[results: Tensor[T], components: Tensor[T]] {.noInit.} =
  ## Principal Component Analysis (PCA)
  ## Inputs:
  ##   - A matrix of shape [Nb of observations, Nb of features]
  ##   - The number of components to keep (default 2D for 2D projection)
  ##
  ## Returns:
  ##   - A tuple of results and components:
  ##     results: a matrix of shape [Nb of observations, Nb of components]
  ##     components: a matrix of shape [Nb of observations, Nb of components] in descending order
  assert x.rank == 2

  let mean_centered = x .- x.mean(axis=0)

  var cov_matrix = newTensorUninit[T]([x.shape[1], x.shape[1]])
  gemm(1.T / T(x.shape[0]-1), mean_centered.transpose, mean_centered, 0, cov_matrix)

  let (_, eigvecs) = cov_matrix.symeig(true, ^nb_components .. ^1) # Note: eigvals/vecs are returned in ascending order

  # [Nb_obs, Nb_feats] * [Nb_feats, Nb_components], don't forget to reorder component in descending order
  result.components = eigvecs[_, ^1..0|-1]
  result.results= mean_centered * result.components

proc pca*[T: SomeReal](x: Tensor[T], principal_axes: Tensor[T]): Tensor[T] {.noInit.} =
  ## Principal Component Analysis (PCA) projection
  ## Inputs:
  ##    - A matrix of shape [Nb of observations, Nb of components]
  ##    - A matrix of shape [Nb of observations, Nb of components] to project on, in descending order
  ##
  ## Returns:
  ##    - A matrix of shape [Nb of observations, Nb of components]
  let mean_centered = x .- x.mean(axis=0)
  result = mean_centered * principal_axes
