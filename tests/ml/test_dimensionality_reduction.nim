# Copyright (c) 2018 Mamy André-Ratsimbazafy and the Arraymancer contributors
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

import ../../src/arraymancer
import std / unittest

proc main() =
  suite "[ML] Dimensionality reduction":
    test "Principal component analysis (PCA)":

      block: # http://www.cs.otago.ac.nz/cosc453/student_tutorials/principal_components.pdf
        let data = [[2.5, 2.4],
                    [0.5, 0.7],
                    [2.2, 2.9],
                    [1.9, 2.2],
                    [3.1, 3.0],
                    [2.3, 2.7],
                    [2.0, 1.6],
                    [1.0, 1.1],
                    [1.5, 1.6],
                    [1.1, 0.9]].toTensor

        let (projected, components) = data.pca(2)

        let expected = [[-0.827970186, -0.175115307],
                        [ 1.77758033,   0.142857227],
                        [-0.992197494,  0.384374989],
                        [-0.274210416,  0.130417207],
                        [-1.67580142,  -0.209498461],
                        [-0.912949103,  0.175282444],
                        [ 0.0991094375,-0.349824698],
                        [ 1.14457216,   0.0464172582],
                        [ 0.438046137,  0.0177646297],
                        [ 1.22382056,  -0.162675287]].toTensor

        for col in 0..<2:
          check:  mean_absolute_error( projected[_, col], expected[_, col]) < 1e-08 or
                  mean_absolute_error(-projected[_, col], expected[_, col]) < 1e-08

        # Projecting the original data with the axes matrix
        let centered = data -. data.mean(axis=0)
        check: projected.mean_absolute_error(centered * components) < 1e-08

      block: # https://www.cgg.com/technicaldocuments/cggv_0000014063.pdf
        let data =  [[ 1.0, -1.0],
                  [ 0.0,  1.0],
                  [-1.0, 0.0]].toTensor

        let (projected, components) = data.pca(2)

        let expected = [[ 2.0,  0.0],
                        [-1.0,  1.0],
                        [-1.0, -1.0]].toTensor / sqrt(2.0)

        for col in 0..<2:
          check:  mean_absolute_error( projected[_, col], expected[_, col]) < 1e-10 or
                  mean_absolute_error(-projected[_, col], expected[_, col]) < 1e-10

        # Projecting the original data with the components matrix
        let centered = data -. data.mean(axis=0)
        check: projected.mean_absolute_error(centered * components) < 1e-08


    test "Principal component analysis (PCA) with full details":
      # import numpy as np
      # from sklearn.decomposition import PCA
      #
      # def test(X):
      #     pca = PCA(n_components=2)
      #
      #
      #     projected = pca.fit_transform(X)
      #
      #     print(f'n_observations: {pca.n_samples_}')
      #     print(f'n_features: {pca.n_features_}')
      #     print(f'n_components: {pca.n_components_}')
      #     print(f'\nprojected:\n{projected}')
      #     print(f'\ncomponents:\n{pca.components_}')
      #     print(f'\nmean:\n{pca.mean_}')
      #     print(f'\nexplained_variance:\n{pca.explained_variance_}')
      #     print(f'\nexplained_variance_ratio:\n{pca.explained_variance_ratio_}')
      #     print(f'\nsingular_values:\n{pca.singular_values_}')
      #     print(f'\nnoise_variance: {pca.noise_variance_}')
      block:
        # X = np.array(
        #       [[2.5, 2.4],
        #       [0.5, 0.7],
        #       [2.2, 2.9],
        #       [1.9, 2.2],
        #       [3.1, 3.0],
        #       [2.3, 2.7],
        #       [2.0, 1.6],
        #       [1.0, 1.1],
        #       [1.5, 1.6],
        #       [1.1, 0.9]])
        let X = [[2.5, 2.4],
                [0.5, 0.7],
                [2.2, 2.9],
                [1.9, 2.2],
                [3.1, 3.0],
                [2.3, 2.7],
                [2.0, 1.6],
                [1.0, 1.1],
                [1.5, 1.6],
                [1.1, 0.9]].toTensor

        let pca_result = pca_detailed(X, n_components = 2)

        let expected = [[-0.827970186, -0.175115307],
                        [ 1.77758033,   0.142857227],
                        [-0.992197494,  0.384374989],
                        [-0.274210416,  0.130417207],
                        [-1.67580142,  -0.209498461],
                        [-0.912949103,  0.175282444],
                        [ 0.0991094375,-0.349824698],
                        [ 1.14457216,   0.0464172582],
                        [ 0.438046137,  0.0177646297],
                        [ 1.22382056,  -0.162675287]].toTensor

        check:
          pca_result.n_observations == 10
          pca_result.n_features == 2
          pca_result.n_components == 2

        # Check the projection
        for col in 0..<2:
          check:  mean_absolute_error( pca_result.projected[_, col], expected[_, col]) < 1e-08 or
                  mean_absolute_error(-pca_result.projected[_, col], expected[_, col]) < 1e-08

        # Check the components matrix by reprojecting
        let centered = X -. X.mean(axis=0)
        check: pca_result.projected.mean_absolute_error(centered * pca_result.components) < 1e-08

        # Check the mean
        let expected_mean = [1.81, 1.91].toTensor()
        check: mean_absolute_error(pca_result.mean, expected_mean) < 1e-08

        # Explained variance
        let expected_explained_variance = [1.28402771, 0.0490834].toTensor()
        check: mean_absolute_error(pca_result.explained_variance, expected_explained_variance) < 1e-08

        # Explained variance ratio
        let expected_explained_variance_ratio = [0.96318131, 0.03681869].toTensor()
        check: mean_absolute_error(pca_result.explained_variance_ratio, expected_explained_variance_ratio) < 1e-08

        # Singular values
        let expected_eigenvals = [3.3994484, 0.66464321].toTensor()
        check: mean_absolute_error(pca_result.singular_values, expected_eigenvals) < 1e-08

        # Noise variance
        let expected_noise = 0.0
        check: absolute_error(expected_noise, expected_noise) < 1e-08

      block:
        # X = np.array(
        #       [[-1, -1],
        #        [-2, -1],
        #        [-3, -2],
        #        [1, 1],
        #        [2, 1],
        #        [3, 2]])
        let X = [[-1, -1],
                [-2, -1],
                [-3, -2],
                [ 1,  1],
                [ 2,  1],
                [ 3,  2]].toTensor().asType(float64)

        let pca_result = pca_detailed(X, n_components = 2)

        let expected = [[ 1.38340578,  0.2935787 ],
                        [ 2.22189802, -0.25133484],
                        [ 3.6053038 ,  0.04224385],
                        [-1.38340578, -0.2935787 ],
                        [-2.22189802,  0.25133484],
                        [-3.6053038 , -0.04224385]].toTensor

        check:
          pca_result.n_observations == 6
          pca_result.n_features == 2
          pca_result.n_components == 2

        # Check the projection
        for col in 0..<2:
          check:  mean_absolute_error( pca_result.projected[_, col], expected[_, col]) < 1e-08 or
                  mean_absolute_error(-pca_result.projected[_, col], expected[_, col]) < 1e-08

        # Check the components matrix by reprojecting
        let centered = X -. X.mean(axis=0)
        check: pca_result.projected.mean_absolute_error(centered * pca_result.components) < 1e-08

        # Check the mean
        let expected_mean = [0.0, 0.0].toTensor()
        check: mean_absolute_error(pca_result.mean, expected_mean) < 1e-08

        # Explained variance
        let expected_explained_variance = [7.93954312, 0.06045688].toTensor()
        check: mean_absolute_error(pca_result.explained_variance, expected_explained_variance) < 1e-08

        # Explained variance ratio
        let expected_explained_variance_ratio = [0.99244289, 0.00755711].toTensor()
        check: mean_absolute_error(pca_result.explained_variance_ratio, expected_explained_variance_ratio) < 1e-08

        # Singular values
        let expected_eigenvals = [6.30061232, 0.54980396].toTensor()
        check: mean_absolute_error(pca_result.singular_values, expected_eigenvals) < 1e-08

        # Noise variance
        let expected_noise = 0.0
        check: absolute_error(expected_noise, expected_noise) < 1e-08

      block:
        # X = np.array(
        #      [[-1, -1, 1],
        #       [-2, -1, 4],
        #       [-3, -2, -10],
        #       [1, 1, 0],
        #       [2, 1, 7],
        #       [3, 2, 3]])
        let X = [[-1, -1, 1],
                [-2, -1, 4],
                [-3, -2, -10],
                [ 1,  1, 0],
                [ 2,  1, 7],
                [ 3,  2, 3]].toTensor().asType(float64)

        let pca_result = pca_detailed(X, n_components = 2)

        let expected = [[ 0.29798775,  1.36325421],
                        [-2.25650538,  3.13955756],
                        [11.41612855, -0.16713955],
                        [ 0.33140736, -1.58301663],
                        [-6.55502622, -0.06288375],
                        [-3.23399207, -2.68977185]].toTensor

        check:
          pca_result.n_observations == 6
          pca_result.n_features == 3
          pca_result.n_components == 2

        # Check the projection
        for col in 0..<2:
          check:  mean_absolute_error( pca_result.projected[_, col], expected[_, col]) < 1e-08 or
                  mean_absolute_error(-pca_result.projected[_, col], expected[_, col]) < 1e-08

        # Check the components matrix by reprojecting
        let centered = X -. X.mean(axis=0)
        check: pca_result.projected.mean_absolute_error(centered * pca_result.components) < 1e-08

        # Check the mean
        let expected_mean = [0.0, 0.0, 0.83333333].toTensor()
        check: mean_absolute_error(pca_result.mean, expected_mean) < 1e-08

        # Explained variance
        let expected_explained_variance = [37.80910168, 4.29759759].toTensor()
        check: mean_absolute_error(pca_result.explained_variance, expected_explained_variance) < 1e-08

        # Explained variance ratio
        let expected_explained_variance_ratio = [0.89665854, 0.10191931].toTensor()
        check: mean_absolute_error(pca_result.explained_variance_ratio, expected_explained_variance_ratio) < 1e-08

        # Singular values
        let expected_eigenvals = [13.74938211, 4.63551378].toTensor()
        check: mean_absolute_error(pca_result.singular_values, expected_eigenvals) < 1e-08

        # Noise variance
        let expected_noise = 0.059967392969596925
        check: absolute_error(expected_noise, expected_noise) < 1e-08


main()
GC_fullCollect()
