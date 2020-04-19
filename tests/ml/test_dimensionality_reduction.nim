# Copyright (c) 2018 Mamy André-Ratsimbazafy and the Arraymancer contributors
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

import ../../src/arraymancer
import unittest, math

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
    #     print(f'n_components: {pca.n_components_}')
    #     print(f'n_features: {pca.n_features_}')
    #     print(f'n_observations: {pca.n_samples_}')
    #     print(f'\nprojected:\n{projected}')
    #     print(f'\ncomponents:\n{pca.components_}')
    #     print(f'\nexplained_variance:\n{pca.explained_variance_}')
    #     print(f'\nexplained_variance_ratio:\n{pca.explained_variance_ratio_}')
    #     print(f'\nmean:\n{pca.mean_}')
    #     print(f'\nsingular_values:\n{pca.singular_values_}')
    #     print(f'\nnoise_variance: {pca.noise_variance_}')
    block:
      # X = np.array(
      #     [[-1, -1],
      #      [-2, -1],
      #      [-3, -2],
      #      [1, 1],
      #      [2, 1],
      #      [3, 2]])
      let X = [[-1, -1],
               [-2, -1],
               [-3, -2],
               [ 1,  1],
               [ 2,  1],
               [ 3,  3]].toTensor().astype(float64)

      let pca_result = pca_detailed(X, n_components = 2)

      echo pca_result
