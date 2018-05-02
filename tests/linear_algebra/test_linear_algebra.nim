# Copyright (c) 2018 the Arraymancer contributors
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).

import ../../src/arraymancer
import unittest, math, fenv

suite "Linear algebra":
  test "Linear equation solver using least squares":
    block: # "Single equation"
           # Example from Numpy documentation
      let A = [
        [0f, 1f],
        [1f, 1f],
        [2f, 1f],
        [3f, 1f]
      ].toTensor

      let y = [-1f, 0.2, 0.9, 2.1].toTensor

      let (solution, residuals, matrix_rank, singular_values) = least_squares_solver(A, y)

      # From Numpy with double precision and `lstsq` function
      let
        expected_sol = [1f, -0.95f].toTensor
        expected_residuals = [0.05f].toTensor
        expected_matrix_rank = 2
        expected_sv = [ 4.10003045f,  1.09075677].toTensor

      check:
        mean_relative_error(solution, expected_sol) < 1e-6
        mean_relative_error(residuals, expected_residuals) < 1e-6
        matrix_rank == expected_matrix_rank
        mean_relative_error(singular_values, expected_sv) < 1e-6

    block: # Example from Eigen
           # https://eigen.tuxfamily.org/dox/group__LeastSquares.html
      let A = [[ 0.68 ,  0.597],
               [-0.211,  0.823],
               [ 0.566, -0.605]].toTensor
      let b = [-0.33, 0.536, -0.444].toTensor

      let (solution, _, _, _) = least_squares_solver(A, b)
      let expected_sol = [-0.67, 0.314].toTensor

      check: mean_relative_error(solution, expected_sol) < 1e-3

    block: # "Multiple independant equations"
           # Example from Intel documentation:
           # https://software.intel.com/sites/products/documentation/doclib/mkl_sa/11/mkl_lapack_examples/dgelsd_ex.c.htm
      let a = [
        [ 0.12,  -8.19,   7.69,  -2.26,  -4.71],
        [-6.91,   2.22,  -5.12,  -9.08,   9.96],
        [-3.33,  -8.94,  -6.72,  -4.40,  -9.98],
        [ 3.97,   3.33,  -2.74,  -7.92,  -3.20]
      ].toTensor

      let b = [
        [ 7.30,   0.47,  -6.28],
        [ 1.33,   6.58,  -3.42],
        [ 2.68,  -1.71,   3.46],
        [-9.62,  -0.79,   0.41]
      ].toTensor

      let (solution, residuals, matrix_rank, singular_values) = least_squares_solver(a, b)

      # From Intel minimum norm solution
      let expected_sol = [[-0.69, -0.24,  0.06],
                          [-0.80, -0.08,  0.21],
                          [ 0.38,  0.12, -0.65],
                          [ 0.29, -0.24,  0.42],
                          [ 0.29,  0.35, -0.30]].toTensor
      # No residuals
      let expected_matrix_rank = 4
      let expected_sv = [ 18.66, 15.99, 10.01, 8.51].toTensor

      check:
        mean_relative_error(solution, expected_sol) < 0.015
        residuals.rank == 0 and residuals.shape[0] == 0 and residuals.strides[0] == 0
        matrix_rank == expected_matrix_rank
        mean_relative_error(singular_values, expected_sv) < 1e-03

  test "Eigenvalues and eigenvector of symmetric matrices":
    block:  # https://software.intel.com/sites/products/documentation/doclib/mkl_sa/11/mkl_lapack_examples/dsyev_ex.c.htm
            # https://pytorch.org/docs/master/torch.html#torch.symeig

      let a =  [[1.96, -6.49, -0.47, -7.20, -0.65],
                [0.00,  3.80, -6.39,  1.50, -6.34],
                [0.00,  0.00,  4.17, -1.51,  2.67],
                [0.00,  0.00,  0.00,  5.70,  1.80],
                [0.00,  0.00,  0.00,  0.00, -7.10]].toTensor

      let expected_val = [-11.0656,  -6.2287,   0.8640,   8.8655,  16.0948].toTensor

      let expected_vec = [[-0.2981, -0.6075,  0.4026, -0.3745,  0.4896],
                          [-0.5078, -0.2880, -0.4066, -0.3572, -0.6053],
                          [-0.0816, -0.3843, -0.6600,  0.5008,  0.3991],
                          [-0.0036, -0.4467,  0.4553,  0.6204, -0.4564],
                          [-0.8041,  0.4480,  0.1725,  0.3108,  0.1622]].toTensor

      let (val, vec) = symeig(a)

      check:
        mean_absolute_error(val, expected_val) < 1e-4
        mean_absolute_error(vec, expected_vec) < 1e-4
