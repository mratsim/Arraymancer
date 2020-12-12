# Copyright (c) 2018 Mamy André-Ratsimbazafy and the Arraymancer contributors
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

import
  ../../tensor, ../../linear_algebra,
  ../../laser/strided_iteration/foreach

proc pca*[T: SomeFloat](
       X: Tensor[T], n_components = 2, center: static bool = true,
       n_oversamples = 5,
       n_power_iters = 2
      ): tuple[projected: Tensor[T], components: Tensor[T]] {.noInit.} =
  ## Principal Component Analysis (PCA)
  ##
  ## Project the input data ``X`` of shape [Observations, Features] into a new coordinate system
  ## where axes (principal components) are in descending order of explained variance of the original ``X`` data
  ## i.e. the first axis explains most of the variance.
  ##
  ## The rotated ``components`` cmatrix can be used to project new observations onto the same base:
  ##   X' * loadings, with X' of shape [Observations', Features].
  ##   X' must be mean centered
  ## Its transposed can be use to reconstruct the original X:
  ##   X ~= projected * components.transpose()
  ##
  ## PCA requires:
  ##   - mean-centered features. This procedure does the centering by default.
  ##     You can pass "center = false", if your preprocessing leads to centering.
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
  ##   - A tuple of PCA projected matrix and principal components matrix:
  ##     projected: a matrix of shape [Nb of observations, Nb of components] in descending order of explained variance
  ##     components: a matrix of shape [Nb of features, Nb of components] to project new data on the same orthogonal basis
  assert X.rank == 2, "Input should be a 2-dimensional matrix"

  when center:
    # TODO: When we center, we could do in-place randomized SVD and save memory from cloning x
    #       but that only happen when the number of components is within 25% of [Observations, Features]
    let X = X -. X.mean(axis=0)

  let (U, S, Vh) = svd_randomized(X, n_components, n_oversamples=n_oversamples, n_power_iters=n_power_iters)
  result.components = Vh.transpose
  result.projected = U *. S.unsqueeze(0) # S sparse diagonal, equivalent to multiplying by a dense diagonal matrix


type PCA_Detailed*[T: SomeFloat] = object
  ## Principal Component Analysis (PCA) object
  ## with full details
  ##
  ## Contains the full PCA details from an input matrix of shape [n_observations, n_features]
  ## - n_observations: The number of observations/samples from an input matrix of shape [n_observations, n_features]
  ## - n_features: The number of features from the input matrix of shape [n_observations, n_features]
  ## - n_components: The number of principal components asked in `pca_detailed`
  ## - projected: The result of the PCA of shape [n_observations, n_components] in descending order of explained variance
  ## - components: a matrix of shape [n_features, n_components] to project new data on the same orthogonal basis
  ## - mean: Per-feature empirical mean, equal to input.mean(axis=0)
  ## - explained_variance: a vector of shape [n_components] in descending order.
  ##     Represents the amount of variance explained by each components
  ##     It is equal to `n_components` largest eigenvalues of the covariance matrix of X.
  ## - explained_variance_ratio: a vector of shape [n_components] in descending order.
  ##     Represents the percentage of variance explained by each components
  ## - singular_values: a vector of shape [n_components] in descending order.
  ##     The singular values corresponding to each components.
  ##     The singular values are equal to the 2-norms of the `n_components` cariables in the lower-dimensional space
  ## - noise_variance: The estimated noise covariance following the Probabilistic PCA model
  ##     from Tipping and Bishop 1999. See "Pattern Recognition and
  ##     Machine Learning" by C. Bishop, 12.2.1 p. 574 or
  ##     http://www.miketipping.com/papers/met-mppca.pdf. It is required to
  ##     compute the estimated data covariance and score samples.##
  ##     Equal to the average of (min(n_features, n_samples) - n_components)
  ##     smallest eigenvalues of the covariance matrix of X.
  ##
  ## The outputs `mean`, `explained_variance`, `explained_variance_ratio`, `singular_values`
  ## are squeezed to 1D and matches the features column vectors
  n_observations*: int
  n_features*: int
  n_components*: int
  projected*: Tensor[T]
  components*: Tensor[T]
  mean*:Tensor[T]
  explained_variance*: Tensor[T]
  explained_variance_ratio*: Tensor[T]
  singular_values*: Tensor[T]
  noise_variance*: T

proc `$`*(pca: PCA_Detailed): string =

  # TODO: use the dup macro
  # TODO: ellipsis in large tensors
  result =  "PCA Detailed: \n"
  result &= "      n_observations: " & $pca.n_observations &
            ", n_features: " & $pca.n_features &
            ", n_components: " & $pca.n_components & '\n'
  result &= "      projected:\n" & $pca.projected & '\n'
  result &= "      components:\n" & $pca.components & '\n'
  result &= "      mean:\n" & $pca.mean & '\n'
  result &= "      explained_variance:\n" & $pca.explained_variance & '\n'
  result &= "      explained_variance_ratio:\n" & $pca.explained_variance_ratio & '\n'
  result &= "      singular_values:\n" & $pca.singular_values & '\n'
  result &= "      noise_variance: " & $pca.noise_variance

proc pca_detailed*[T: SomeFloat](
       X: Tensor[T], n_components = 2, center: static bool = true,
       n_oversamples = 5,
       n_power_iters = 2
      ): PCA_Detailed[T] {.noInit.} =
  ## Principal Component Analysis (PCA) with full details
  ##
  ## Project the input data ``X`` of shape [Observations, Features] into a new coordinate system
  ## where axes (principal components) are in descending order of explained variance of the original ``X`` data
  ## i.e. the first axis explains most of the variance.
  ##
  ## The rotated ``components`` cmatrix can be used to project new observations onto the same base:
  ##   X' * loadings, with X' of shape [Observations', Features].
  ##   X' must be mean centered
  ## Its transposed can be use to reconstruct the original X:
  ##   X ~= projected * components.transpose()
  ##
  ## PCA requires:
  ##   - mean-centered features. This procedure does the centering by default.
  ##     You can pass "center = false", if your preprocessing leads to centering.
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
  ## Returns a "Principal Component Analysis" object with the following fields
  ## - n_observations: The number of observations/samples from an input matrix of shape [n_observations, n_features]
  ## - n_features: The number of features from the input matrix of shape [n_observations, n_features]
  ## - n_components: The number of principal components asked in `pca_detailed`
  ## - projected: The result of the PCA of shape [n_observations, n_components] in descending order of explained variance
  ## - components: a matrix of shape [n_features, n_components] to project new data on the same orthogonal basis
  ## - mean: Per-feature empirical mean, equal to input.mean(axis=0)
  ## - explained_variance: a vector of shape [n_components] in descending order.
  ##     Represents the amount of variance explained by each components
  ##     It is equal to `n_components` largest eigenvalues of the covariance matrix of X.
  ## - explained_variance_ratio: a vector of shape [n_components] in descending order.
  ##     Represents the percentage of variance explained by each components
  ## - singular_values: a vector of shape [n_components] in descending order.
  ##     The singular values corresponding to each components.
  ##     The singular values are equal to the 2-norms of the `n_components` cariables in the lower-dimensional space
  ## - noise_variance: The estimated noise covariance following the Probabilistic PCA model
  ##     from Tipping and Bishop 1999. See "Pattern Recognition and
  ##     Machine Learning" by C. Bishop, 12.2.1 p. 574 or
  ##     http://www.miketipping.com/papers/met-mppca.pdf. It is required to
  ##     compute the estimated data covariance and score samples.##
  ##     Equal to the average of (min(n_features, n_samples) - n_components)
  ##     smallest eigenvalues of the covariance matrix of X.
  ##
  ## The outputs `mean`, `explained_variance`, `explained_variance_ratio`, `singular_values`
  ## are squeezed to 1D and matches the features column vectors
  assert X.rank == 2, "Input should be a 2-dimensional matrix"

  result.n_observations = X.shape[0]
  result.n_features = X.shape[1]
  result.n_components = n_components

  result.mean = X.mean(axis=0)

  when center:
    # TODO: When we center, we could do in-place randomized SVD and save memory from cloning x
    #       but that only happen when the number of components is within 25% of [Observations, Features]
    let X = X -. result.mean

  result.mean = result.mean.squeeze(axis = 0)

  let (U, S, Vh) = svd_randomized(X, n_components, n_oversamples=n_oversamples, n_power_iters=n_power_iters)
  result.components = Vh.transpose
  result.projected = U *. S.unsqueeze(0) # S sparse diagonal, equivalent to multiplying by a dense diagonal matrix

  # Variance explained by Singular Values
  let bessel_correction = T(result.n_observations - 1)
  result.explained_variance = newTensorUninit[T](S.shape)
  forEach ev in result.explained_variance,
          s in S:
    ev = s*s / bessel_correction

  # Since we are using SVD truncated to `n_components` we need to
  # refer back to the original matrix for total variance
  # We assume that the divisor is (N-1) (unbiaised ~ Bessel correction)
  let total_variance = X.variance(axis = 0) # assumes unbiaised
  let sum_total_var = total_variance.sum()
  result.explained_variance_ratio = result.explained_variance /. sum_total_var

  result.singular_values = S

  # Noise covariance
  #      "Pattern Recognition and
  #     Machine Learning" by C. Bishop, 12.2.1 p. 574 or
  #     http://www.miketipping.com/papers/met-mppca.pdf
  # Equation 12.46
  if result.n_components < min(result.n_features, result.n_observations):
    result.noise_variance = sum_total_var - result.explained_variance.sum()
    result.noise_variance /= T(min(result.n_features, result.n_observations) - n_components)
  else:
    result.noise_variance = 0
