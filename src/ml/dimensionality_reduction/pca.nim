# Copyright (c) 2018 Mamy André-Ratsimbazafy and the Arraymancer contributors
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

import
  ../../tensor/tensor, ../../linear_algebra/linear_algebra


proc pca*[T: SomeFloat](
       x: Tensor[T], n_components = 2, center: static bool = true,
       n_oversamples = 5,
       n_power_iters = 2
      ): tuple[scores: Tensor[T], loadings: Tensor[T]] {.noInit.} =
  ## Principal Component Analysis (PCA)
  ##
  ## Project the input data ``X`` of shape [Observations, Features] into a new coordinate system
  ## where axis are in descending order of explained variance of the original ``X`` data
  ##
  ## Projection is called ``scores`` and the project values are linearly uncorrelated.
  ## The rotation matrix ``loadings`` can be used to project new observations onto the same base:
  ##   X' * loadings, with X' of shape [Observations', Features]
  ## Its transposed can be use to reconstruct the original X:
  ##   X ~= scores * loadings.transpose()
  ##
  ## PCA requires:
  ##   - mean-centered features. This procedure does the centering by default.
  ##   - Features of the same scale/amplitude. Some alternatives include
  ##     min-max scaling, mean normalization, standardization (mean = 0 and unit variance),
  ##     rescaling column to unit-length.
  ##
  ## Note: PCA without centering is also called truncated SVD,
  ##       which is useful when centering is costly, for example
  ##       in the case of sparse matrices from parsing text.
  ##
  ## Inputs:
  ##   - A matrix of shape [Nb of observations, Nb of features]
  ##   - The number of components to keep (default 2D for 2D projection)
  ##
  ## Returns:
  ##   - A tuple of PCA scores (projected matrix) and loadings (rotation matrix):
  ##     scores: a matrix of shape [Nb of observations, Nb of components] in descending order of explained variance
  ##     loadings: a matrix of shape [Nb of features, Nb of components] in descending order of explained variance
  assert x.rank == 2

  when center:
    # TODO: When we center, we could do in-place randomized SVD and save memory from cloning x
    #       but that only happen when the number of components is within 25% of [Observations, Features]
    let x = x .- x.mean(axis=0)

  let (U, S, Vh) = svd_randomized(x, n_components)
  result.scores = U .* S.unsqueeze(0)
  result.loadings = Vh
