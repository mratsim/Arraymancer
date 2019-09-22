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
        mean_relative_error(residuals, expected_residuals) < 2e-6 # Due to parallelism hazards this sometimes go over 1e-6 on Travis
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
    # Note: Functions should return a unit vector (norm == 1).
    #       But, if v is an eigen vector, any λv is also an eigen vector especially for λ = -1.
    #       changing library implementation will create failures if they return a vector of the opposite sign.
    #       (even though it is also a unit vector)

    block:  # https://software.intel.com/sites/products/documentation/doclib/mkl_sa/11/mkl_lapack_examples/dsyev_ex.c.htm
            # https://pytorch.org/docs/master/torch.html#torch.symeig

      let a =  [[ 1.96, -6.49, -0.47, -7.20, -0.65],
                [-6.49,  3.80, -6.39,  1.50, -6.34],
                [-0.47, -6.39,  4.17, -1.51,  2.67],
                [-7.20,  1.50, -1.51,  5.70,  1.80],
                [-0.65, -6.34,  2.67,  1.80, -7.10]].toTensor

      let expected_val = [-11.0656,  -6.2287,   0.8640,   8.8655,  16.0948].toTensor

      let expected_vec = [[-0.2981, -0.6075,  0.4026, -0.3745,  0.4896],
                          [-0.5078, -0.2880, -0.4066, -0.3572, -0.6053],
                          [-0.0816, -0.3843, -0.6600,  0.5008,  0.3991],
                          [-0.0036, -0.4467,  0.4553,  0.6204, -0.4564],
                          [-0.8041,  0.4480,  0.1725,  0.3108,  0.1622]].toTensor

      let (val, vec) = symeig(a, true)

      check: mean_absolute_error(val, expected_val) < 1e-4

      for col in 0..<5:
        check:  mean_absolute_error( vec[_, col], expected_vec[_, col]) < 1e-4 or
                mean_absolute_error(-vec[_, col], expected_vec[_, col]) < 1e-4

    block: # p15 of http://www.cs.otago.ac.nz/cosc453/student_tutorials/principal_components.pdf

      let a =  [[0.616555556'f32, 0.615444444],
                [0.615444444'f32, 0.716555556]].toTensor

      let expected_val = [0.0490833989'f32, 1.28402771].toTensor

      # Note: here the 1st vec is returned positive by Fortran (both are correct eigenvec)
      let expected_vec = [[-0.735178656'f32, -0.677873399],
                          [ 0.677873399'f32, -0.735178656]].toTensor

      let (val, vec) = symeig(a, true)

      check: mean_absolute_error(val, expected_val) < 1e-7

      for col in 0..<2:
        check:  mean_absolute_error( vec[_, col], expected_vec[_, col]) < 1e-11 or
                mean_absolute_error(-vec[_, col], expected_vec[_, col]) < 1e-11

    block: # https://software.intel.com/sites/products/documentation/doclib/mkl_sa/11/mkl_lapack_examples/dsyevd_ex.c.htm

      let a =  [[ 6.39,  0.13, -8.23,  5.71, -3.18],
                [ 0.13,  8.37, -4.46, -6.10,  7.21],
                [-8.23, -4.46, -9.58, -9.25, -7.42],
                [ 5.71, -6.10, -9.25,  3.72,  8.54],
                [-3.18,  7.21, -7.42,  8.54,  2.51]].toTensor

      let expected_val = [-17.44, -11.96, 6.72, 14.25, 19.84].toTensor

      let expected_vec = [[-0.26,  0.31, -0.74,  0.33,  0.42],
                          [-0.17, -0.39, -0.38, -0.80,  0.16],
                          [-0.89,  0.04,  0.09,  0.03, -0.45],
                          [-0.29, -0.59,  0.34,  0.31,  0.60],
                          [-0.19,  0.63,  0.44, -0.38,  0.48]].toTensor

      let (val, vec) = symeig(a, true)

      check: mean_absolute_error(val, expected_val) < 1e-2

      for col in 0..<5:
        check:  mean_absolute_error( vec[_, col], expected_vec[_, col]) < 1e-2 or
                mean_absolute_error(-vec[_, col], expected_vec[_, col]) < 1e-2

    block: # https://software.intel.com/sites/products/documentation/doclib/mkl_sa/11/mkl_lapack_examples/dsyevr_ex.c.htm

      let a =  [[ 0.67, -0.20,  0.19, -1.06,  0.46],
                [-0.20,  3.82, -0.13,  1.06, -0.48],
                [ 0.19, -0.13,  3.27,  0.11,  1.10],
                [-1.06,  1.06,  0.11,  5.86, -0.98],
                [ 0.46, -0.48,  1.10, -0.98,  3.54]].toTensor

      let selected_val = [0.43, 2.14, 3.37].toTensor

      let selected_vec = [[-0.98, -0.01, -0.08],
                          [ 0.01,  0.02, -0.93],
                          [ 0.04, -0.69, -0.07],
                          [-0.18,  0.19,  0.31],
                          [ 0.07,  0.69, -0.13]].toTensor

      let (val, vec) = symeig(a, true, 0..2)

      check: mean_absolute_error(val, selected_val) < 1e-1

      for col in 0..<3:
        check:  mean_absolute_error( vec[_, col], selected_vec[_, col]) < 1e-2 or
                mean_absolute_error(-vec[_, col], selected_vec[_, col]) < 1e-2

    block: # Check that input is not overwritten

      let a, b =[[ 0.67, -0.20,  0.19, -1.06,  0.46],
                [-0.20,  3.82, -0.13,  1.06, -0.48],
                [ 0.19, -0.13,  3.27,  0.11,  1.10],
                [-1.06,  1.06,  0.11,  5.86, -0.98],
                [ 0.46, -0.48,  1.10, -0.98,  3.54]].toTensor

      discard symeig(a, true, 0..2)
      check: a == b

  test "QR Decomposition":
    block: # From wikipedia https://en.wikipedia.org/wiki/QR_decomposition
      let a = [[12.0, -51.0, 4.0],
              [ 6.0, 167.0, -68.0],
              [-4.0,  24.0, -41.0]].toTensor()

      let (q, r) = qr(a)

      block: # Sanity checks
        # A = QR
        let qr = q * r
        check: mean_absolute_error(a, qr) < 1e-8
        # TODO: r is triangular

      block: # vs NumPy implementation. Note that
             # decomposition are not unique, there can be a factor +1/-1.
        # import numpy as np
        # a = np.array([[12,-51,4],[6,167,-68],[-4,24,-41]])
        # q, r = np.linalg.qr(a)
        # print(q)
        # print(r)
        let np_q = [[-0.85714286,  0.39428571,  0.33142857],
                   [-0.42857143, -0.90285714, -0.03428571],
                   [ 0.28571429, -0.17142857,  0.94285714]].toTensor()
        let np_r = [[ -14.0,  -21.0,   14.0],
                    [   0.0, -175.0,   70.0],
                    [   0.0,    0.0,  -35.0]].toTensor()

        check:
          mean_absolute_error(q, np_q) < 1e-8
          mean_absolute_error(r, np_r) < 1e-8

    block: # M > N, from Numpy https://docs.scipy.org/doc/numpy-1.15.1/reference/generated/numpy.linalg.qr.html
      let a = [[0.0, 1.0],
               [1.0, 1.0],
               [1.0, 1.0],
               [2.0, 1.0]].toTensor()

      let (q, r) = qr(a)

      block: # Sanity checks
        # A = QR
        let qr = q * r
        check: mean_absolute_error(a, qr) < 1e-8
        # TODO: r is triangular

      block: # vs NumPy implementation. Note that
             # decomposition are not unique, there can be a factor +1/-1.
        # import numpy as np
        # a = np.array([[0, 1], [1, 1], [1, 1], [2, 1]])
        # q, r = np.linalg.qr(a)
        # print(q)
        # print(r)
        let np_q = [[ 0.0       ,  0.8660254 ],
                    [-0.40824829,  0.28867513],
                    [-0.40824829,  0.28867513],
                    [-0.81649658, -0.28867513]].toTensor()
        let np_r = [[-2.44948974, -1.63299316],
                    [ 0.0       ,  1.15470054]].toTensor()

        check:
          mean_absolute_error(q, np_q) < 1e-8
          mean_absolute_error(r, np_r) < 1e-8

    block: # M < N
      let a = [[0.0, 1.0, 1.0, 1.0],
               [1.0, 1.0, 2.0, 1.0]].toTensor()

      let (q, r) = qr(a)

      block: # Sanity checks
        # A = QR
        let qr = q * r
        check: mean_absolute_error(a, qr) < 1e-8
        # TODO: r is triangular

      block: # vs NumPy implementation. Note that
             # decomposition are not unique, there can be a factor +1/-1.
        # import numpy as np
        # a = np.array([[0, 1, 1, 1], [1, 1, 2, 1]])
        # q, r = np.linalg.qr(a)
        # print(q)
        # print(r)
        let np_q = [[ 0.0, -1.0],
                    [-1.0,  0.0]].toTensor()
        let np_r = [[-1.0, -1.0, -2.0, -1.0],
                    [ 0.0, -1.0, -1.0, -1.0]].toTensor()

        check:
          mean_absolute_error(q, np_q) < 1e-8
          mean_absolute_error(r, np_r) < 1e-8
