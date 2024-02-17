# Copyright (c) 2018 the Arraymancer contributors
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).

import ../../src/arraymancer
import unittest, math
import std/strformat

proc main() =
  suite "Linear algebra":
    test "Hilbert matrix":
      block:
        const N = 3
        let a = hilbert(N, float64)
        for i in 1 .. N:
          for j in 1 .. N:
            check:
              a[i-1, j-1] == 1 / (i.float64 + j.float64 - 1)
      block:
        const N = 100
        let a = hilbert(N, float64)
        for i in 1 .. N:
          for j in 1 .. N:
            check:
              a[i-1, j-1] == 1 / (i.float64 + j.float64 - 1)

    test "Vandermonde matrix":
      let x = [1, 2, 3, 4].toTensor
      let A = vandermonde(x, 3)
      for i, ax in enumerateAxis(A, axis = 0):
        var order = 0
        var el = x[i]
        let axSq = ax.squeeze
        for j in 0 ..< axSq.size:
          check axSq[j] == pow(el.float, order.float)
          inc order
      let vandermonde_3x3 = vandermonde(arange(3), 3)
      let vander_3x3 = vander(arange(3), 3)
      check:
        vandermonde_3x3 == vandermonde(3)
        vandermonde_3x3 == vandermonde(arange(3))
        vandermonde_3x3 == vandermonde(arange(3), arange(3))
        vandermonde_3x3 == vander_3x3[_, _|-1]
        vandermonde_3x3 == vander(arange(3), 3, increasing=true)

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
          solution.mean_relative_error(expected_sol) < 1e-6
          residuals.mean_relative_error(expected_residuals) < 2e-6 # Due to parallelism hazards this sometimes go over 1e-6 on Travis
          matrix_rank == expected_matrix_rank
          singular_values.mean_relative_error(expected_sv) < 1e-6

      block: # Example from Eigen
            # https://eigen.tuxfamily.org/dox/group__LeastSquares.html
        let A = [[ 0.68 ,  0.597],
                [-0.211,  0.823],
                [ 0.566, -0.605]].toTensor
        let b = [-0.33, 0.536, -0.444].toTensor

        let (solution, _, _, _) = least_squares_solver(A, b)
        let expected_sol = [-0.67, 0.314].toTensor

        check: solution.mean_relative_error(expected_sol) < 1e-3

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
          solution.mean_relative_error(expected_sol) < 0.015
          residuals.rank == 0 and residuals.shape[0] == 0 and residuals.strides[0] == 0
          matrix_rank == expected_matrix_rank
          singular_values.mean_relative_error(expected_sv) < 1e-03

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

        let (val, vec) = symeig(a, true, 'U')

        check: val.mean_absolute_error(expected_val) < 1e-4

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

        let (val, vec) = symeig(a, true, 'U')

        check: val.mean_absolute_error(expected_val) < 1e-7

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

        let (val, vec) = symeig(a, true, 'U')

        check: val.mean_absolute_error(expected_val) < 1e-2

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

        let (val, vec) = symeig(a, true, 'U', 0..2)

        check: val.mean_absolute_error(selected_val) < 1e-1

        for col in 0..<3:
          check:  mean_absolute_error( vec[_, col], selected_vec[_, col]) < 1e-2 or
                  mean_absolute_error(-vec[_, col], selected_vec[_, col]) < 1e-2

      block: # Check that input is not overwritten

        let a, b =[[ 0.67, -0.20,  0.19, -1.06,  0.46],
                  [-0.20,  3.82, -0.13,  1.06, -0.48],
                  [ 0.19, -0.13,  3.27,  0.11,  1.10],
                  [-1.06,  1.06,  0.11,  5.86, -0.98],
                  [ 0.46, -0.48,  1.10, -0.98,  3.54]].toTensor

        discard symeig(a, true, 'U', 0..2)
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
          check: a.mean_absolute_error(qr) < 1e-8
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
            q.mean_absolute_error(np_q) < 1e-8
            r.mean_absolute_error(np_r) < 1e-8

      block: # M > N, from Numpy https://docs.scipy.org/doc/numpy-1.15.1/reference/generated/numpy.linalg.qr.html
        let a = [[0.0, 1.0],
                [1.0, 1.0],
                [1.0, 1.0],
                [2.0, 1.0]].toTensor()

        let (q, r) = qr(a)

        block: # Sanity checks
          # A = QR
          let qr = q * r
          check: a.mean_absolute_error(qr) < 1e-8
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
            q.mean_absolute_error(np_q) < 1e-8
            r.mean_absolute_error(np_r) < 1e-8

      block: # M < N
        let a = [[0.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 2.0, 1.0]].toTensor()

        let (q, r) = qr(a)

        block: # Sanity checks
          # A = QR
          let qr = q * r
          check: a.mean_absolute_error(qr) < 1e-8
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
            q.mean_absolute_error(np_q) < 1e-8
            r.mean_absolute_error(np_r) < 1e-8

    test "LU Factorization":
      block: # M > N
        # import numpy as np
        # from scipy.linalg import lu
        # np.set_printoptions(suppress=True) # Don't use scientific notation
        # a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
        # pl, u = lu(a, permute_l = True)
        # print(pl)
        # print(u)

        let a = [[ 1.0, 2, 3],
                [ 4.0, 5, 6],
                [ 7.0, 8, 9],
                [10.0,11,12]].toTensor()

        let expected_pl = [[0.1, 1         , 0],
                          [0.4, 0.66666667, 0],
                          [0.7, 0.33333333, 1],
                          [1.0, 0         , 0]].toTensor()
        let expected_u  = [[10.0, 11.0, 12.0],
                          [ 0.0,  0.9,  1.8],
                          [ 0.0,  0.0,  0.0]].toTensor()

        let (PL, U) = lu_permuted(a)
        check:
          PL.mean_absolute_error(expected_pl) < 1e-8
          U.mean_absolute_error(expected_u) < 1e-8

      block: # M < N
        # import numpy as np
        # from scipy.linalg import lu
        # np.set_printoptions(suppress=True) # Don't use scientific notation
        # a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
        # pl, u = lu(a, permute_l = True)
        # print(pl)
        # print(u)

        let a = [[ 1.0,  2,  3,  4],
                [ 5.0,  6,  7,  8],
                [ 9.0, 10, 11, 12]].toTensor()

        let expected_pl = [[0.11111111, 1.0, 0],
                          [0.55555555, 0.5, 1],
                          [1.0, 0         , 0]].toTensor()
        let expected_u  = [[ 9.0, 10.0       , 11.0       , 12.0       ],
                          [ 0.0,  0.88888889,  1.77777778,  2.66666667],
                          [ 0.0,  0.0       , -0.0       , -0.0       ]].toTensor()

        let (PL, U) = lu_permuted(a)
        check:
          PL.mean_absolute_error(expected_pl) < 1e-8
          U.mean_absolute_error(expected_u) < 1e-8

    test "Singular Value Decomposition (SVD)":
      block: # From https://software.intel.com/sites/products/documentation/doclib/mkl_sa/11/mkl_lapack_examples/sgesdd_ex.c.htm
        let a = [[  7.52, -1.10, -7.95,  1.08],
                [ -0.76,  0.62,  9.34, -7.10],
                [  5.13,  6.62, -5.66,  0.87],
                [ -4.75,  8.52,  5.75,  5.30],
                [  1.33,  4.91, -5.49, -3.52],
                [ -2.40, -6.77,  2.34,  3.95]].toTensor()

        let expected_U =  [[-0.57,  0.18,  0.01,  0.53],
                          [ 0.46, -0.11, -0.72,  0.42],
                          [-0.45, -0.41,  0.00,  0.36],
                          [ 0.33, -0.69,  0.49,  0.19],
                          [-0.32, -0.31, -0.28, -0.61],
                          [ 0.21,  0.46,  0.39,  0.09]].toTensor()

        let expected_S = [18.37, 13.63, 10.85, 4.49].toTensor()

        let expected_Vh = [[-0.52, -0.12,  0.85, -0.03],
                          [ 0.08, -0.99, -0.09, -0.01],
                          [-0.28, -0.02, -0.14,  0.95],
                          [ 0.81,  0.01,  0.50,  0.31]].toTensor()

        let (U, S, Vh) = svd(a)

        check:
          U.mean_absolute_error(expected_U) < 1e-2
          S.mean_absolute_error(expected_S) < 1e-2
          Vh.mean_absolute_error(expected_Vh) < 1e-2

    test "SVD for complex" :

        let a = [[  7.52, -1.10, -7.95,  1.08],
                [ -0.76,  0.62,  9.34, -7.10],
                [  5.13,  6.62, -5.66,  0.87],
                [ -4.75,  8.52,  5.75,  5.30],
                [  1.33,  4.91, -5.49, -3.52],
                [ -2.40, -6.77,  2.34,  3.95]].toTensor()
        # just to have _something_ complex:
        let c = a.map_inline(complex(x, 1.0-x))

        # these are obviously wrong
        let expected_U = [[complex(-0.4138926243038071, 0.4037367895674472), complex(0.08389522779838721, -0.1932656628436422), complex(0.01621459048974143, 0.03639076395179549), complex(0.3807114039382682, -0.1271694919561013)],
          [complex(0.327296995907484, -0.3076628167233709), complex(-0.1104077088428707, 0.03141697074805642), complex(-0.5465602539891018, 0.5052813715527299), complex(0.3005144202521765, -0.05732671988915428)],
          [complex(-0.3147019801282679, 0.323121874032496), complex(-0.1855451930080691, 0.3462326420058709), complex(-0.004992387574187037, -0.006241797578430084), complex(0.2614640945138078, -0.1190718866934908)],
          [complex(0.256342930610358, -0.2185199745710323), complex(-0.3075372806738432, 0.5822430422779921), complex(0.351838533491184, -0.3299477382698799), complex(0.1351035392947972, 0.06304842217810554)],
          [complex(-0.2289816303001176, 0.2308419507029495), complex(-0.178259697767442, 0.2497440202894774), complex(-0.2278471684980899, 0.1852158345820878), complex(-0.4362893246683154, 0.5727511306523236)],
          [complex(0.1445257449183605, -0.1428065988449038), complex(0.1992485425375248, -0.4697811283452841), complex(0.3070049859786909, -0.1765909880989579), complex(0.06702805541734494, 0.3429822563170669)]].toTensor()

        let expected_S = [26.0332, 18.8352, 15.4257, 6.33257].toTensor()

        let expected_Vh = [[complex(-0.5059390297870948, 0.0), complex(-0.1074686670590907, -0.02413230677827292), complex(0.8549349975704502, -0.0252434139363754), complex(-0.01487079140603485, -0.01102701470044396)],
          [complex(0.1094836347283804, 0.0), complex(-0.9332707465643947, -0.3268707297146123), complex(-0.06225502820686645, -0.01459498699291252), complex(0.04372857968946522, -0.06460404418932869)],
          [complex(-0.264512197766454, 0.0), complex(0.006428570013845377, 0.06218179833088577), complex(-0.1385452152499669, -0.001896615870859991), complex(0.9490066561220873, -0.07945682785152652)],
          [complex(0.8136782712457769, 0.0), complex(0.06084187977910822, 0.04919065096065829), complex(0.4949302504756131, -0.01434890739447998), complex(0.2933746262122985, -0.0239937729634309)]].toTensor()

        let (U, S, Vh) = svd(c)

        check:
          U.mean_absolute_error(expected_U) < 1e-2
          S.mean_absolute_error(expected_S) < 1e-2
          Vh.mean_absolute_error(expected_Vh) < 1e-2

    test "Randomized SVD":
      # TODO: tests using spectral norm, see fbpca python package
      # Using Hilbert matrix, see ../manual_checks/randomized_svd.py
      block:
        # Going through the **full** SVD codepath
        # n_components + n_over_samples >= observations
        const
          Observations = 10
          Features = 4000
          N = max(Observations, Features)
          k = 7

        let H = hilbert(N, float64)[0..<Observations, 0..<Features]
        let (U, S, Vh) = svd_randomized(H, n_components=k, n_oversamples=5, n_power_iters=2)

        let expected_S = [1.90675907e+00, 4.86476625e-01, 7.52734238e-02, 8.84829787e-03, 7.86824889e-04, 3.71028924e-05, 1.74631562e-06].toTensor()

        check:
          U.shape[0] == H.shape[0]
          U.shape[1] == k
          S.mean_absolute_error(expected_S) < 1.5e-5
          Vh.shape[0] == k
          Vh.shape[1] == H.shape[1]

        let reconstructed = (U *. S.unsqueeze(0)) * Vh
        check: H.mean_absolute_error(reconstructed) < 1e-2

      block:
        # Going through the **full** SVD codepath
        # n_components + n_over_samples >= features
        # Ensure that m > n / m < n logic is working fine
        const
          Observations = 4000
          Features = 10
          N = max(Observations, Features)
          k = 7

        let H = hilbert(N, float64)[0..<Observations, 0..<Features]
        let (U, S, Vh) = svd_randomized(H, n_components=k, n_oversamples=5, n_power_iters=2)

        let expected_S = [1.90675907e+00, 4.86476625e-01, 7.52734238e-02, 8.84829787e-03, 7.86824889e-04, 3.71028924e-05, 1.74631562e-06].toTensor()

        check:
          U.shape[0] == H.shape[0]
          U.shape[1] == k
          S.mean_absolute_error(expected_S) < 1.5e-5
          Vh.shape[0] == k
          Vh.shape[1] == H.shape[1]

        let reconstructed = (U *. S.unsqueeze(0)) * Vh
        check: H.mean_absolute_error(reconstructed) < 1e-2

      block:
        # Going through the **randomized** SVD codepath
        const
          Observations = 200
          Features = 4000
          N = max(Observations, Features)
          k = 7

        let H = hilbert(N, float64)[0..<Observations, 0..<Features]
        let (U, S, Vh) = svd_randomized(H, n_components=k, n_oversamples=5, n_power_iters=2)

        let expected_S = [2.34837630e+00, 1.07682981e+00, 3.72203282e-01, 1.12672683e-01, 3.16910267e-02, 8.45798286e-03, 2.10165880e-03].toTensor()

        check:
          U.shape[0] == H.shape[0]
          U.shape[1] == k
          S.mean_absolute_error(expected_S) < 1.5e-5
          Vh.shape[0] == k
          Vh.shape[1] == H.shape[1]

        let reconstructed = (U *. S.unsqueeze(0)) * Vh
        check: H.mean_absolute_error(reconstructed) < 1e-2

      block:
        # Going through the **randomized** SVD codepath
        # Ensure that m > n / m < n logic is working fine
        const
          Observations = 4000
          Features = 200
          N = max(Observations, Features)
          k = 7

        let H = hilbert(N, float64)[0..<Observations, 0..<Features]
        let (U, S, Vh) = svd_randomized(H, n_components=k, n_oversamples=5, n_power_iters=2)

        let expected_S = [2.34837630e+00, 1.07682981e+00, 3.72203282e-01, 1.12672683e-01, 3.16910267e-02, 8.45798286e-03, 2.10165880e-03].toTensor()

        check:
          U.shape[0] == H.shape[0]
          U.shape[1] == k
          S.mean_absolute_error(expected_S) < 1.5e-5
          Vh.shape[0] == k
          Vh.shape[1] == H.shape[1]

        let reconstructed = (U *. S.unsqueeze(0)) * Vh
        check: H.mean_absolute_error(reconstructed) < 1e-2

    test "Pseudo-Inverse (Pinv)":
      block:
        let a = [[ 7.52, -1.10, -7.95,  1.08],
                [ -0.76,  0.62,  9.34, -7.10],
                [  5.13,  6.62, -5.66,  0.87],
                [ -4.75,  8.52,  5.75,  5.30],
                [  1.33,  4.91, -5.49, -3.52],
                [ -2.40, -6.77,  2.34,  3.95]].toTensor()

        let expected_a_pinv = [[0.11181,    0.0799877,  0.075215,  0.0069947, -0.0949072,  0.00267723],
                              [-0.00777931, 0.00729835, 0.0340491, 0.0477368,  0.0235042, -0.0353603 ],
                              [ 0.0315948,  0.0781534,  0.0227237, 0.0345276, -0.0772229,  0.0116902 ],
                              [ 0.0381356, -0.0348393,  0.0267878, 0.0563091, -0.0662466,  0.0396261 ]].toTensor()

        let a_pinv = pinv(a)
        let should_be_identity = a_pinv * a
        let identity = [[1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0]].toTensor()

        check:
          a_pinv.mean_absolute_error(expected_a_pinv) < 1e-2
          should_be_identity.mean_absolute_error(identity) < 1e-2

    test "Solve linear equations general matrix":
      block:
        # From: https://software.intel.com/sites/products/documentation/doclib/mkl_sa/11/mkl_lapack_examples/dgesv_ex.c.htm
        let a = [[ 6.80, -6.05, -0.45,  8.32, -9.67],
                [-2.11, -3.30,  2.58,  2.71, -5.14],
                [ 5.66,  5.36, -2.70,  4.35, -7.26],
                [ 5.97, -4.44,  0.27, -7.17,  6.08],
                [ 8.23,  1.08,  9.04,  2.14, -6.87]].toTensor

        let b = [[ 4.02, -1.56,  9.81],
                [ 6.19,  4.00, -4.09],
                [-8.22, -8.67, -4.57],
                [-7.57,  1.75, -8.61],
                [-3.03,  2.86,  8.99]].toTensor

        let x_known = [[-0.80, -0.39,  0.96],
                      [-0.70, -0.55,  0.22],
                      [ 0.59,  0.84,  1.90],
                      [ 1.32, -0.10,  5.36],
                      [ 0.57,  0.11,  4.04]].toTensor

        let x = solve(a, b)
        check x.mean_absolute_error(x_known) < 0.01


main()
GC_fullCollect()
