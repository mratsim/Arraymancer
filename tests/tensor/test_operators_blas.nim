# Copyright 2017 the Arraymancer contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import ../../src/arraymancer
import unittest, sugar
import complex except Complex64, Complex32

suite "BLAS (Basic Linear Algebra Subprograms)":
  test "GEMM - General Matrix to Matrix Multiplication":
    ## TODO: test with slices
    let a = [[1.0,2,3],
             [4.0,5,6]].toTensor()

    let b = [[7.0,  8],
             [9.0, 10],
             [11.0,12]].toTensor()

    let ab = [[ 58.0, 64],
              [139.0,154]].toTensor()

    let a_c = a.astype(Complex[float64])
    let b_c = b.astype(Complex[float64])
    let ab_c = ab.astype(Complex[float64])

    check: a * b == ab
    check: a_c * b_c == ab_c

    # example from http://www.intmath.com/matrices-determinants/matrix-multiplication-examples.php
    # (M x K) * (K x N) with M < N
    let u = [[-2,-3,-1],
             [ 3, 0, 4]].toTensor()
    let v = [[ 1, 5, 2,-1],
             [-3, 0, 3, 4],
             [ 6,-2, 7,-4]].toTensor()

    let uv = [[ 1,-8,-20, -6],
              [27, 7, 34,-19]].toTensor()

    check: u.astype(float32) * v.astype(float32) == uv.astype(float32)
    check: u.astype(Complex[float32]) * v.astype(Complex[float32]) == uv.astype(Complex[float32])

    # from http://www.calcul.com/show/calculator/matrix-multiplication_;5;5;5;5?matrix1=[[%225%22,%226%22,%225%22,%228%22],[%228%22,%222%22,%228%22,%228%22],[%220%22,%225%22,%224%22,%220%22],[%224%22,%220%22,%225%22,%226%22],[%224%22,%225%22,%220%22,%223%22]]&matrix2=[[%225%22,%223%22,%226%22,%220%22],[%225%22,%222%22,%223%22,%223%22],[%228%22,%228%22,%222%22,%220%22],[%227%22,%227%22,%220%22,%220%22]]&operator=*
    # (M x K) * (K x N) with M > N and M > block-size (4x4)
    let m1 = [[5,6,5,8],
              [8,2,8,8],
              [0,5,4,0],
              [4,0,5,6],
              [4,5,0,3]].toTensor()
    let m2 = [[5,3,6,0],
              [5,2,3,3],
              [8,8,2,0],
              [7,7,0,0]].toTensor()

    let m1m2 = [[151,123,58,18],
                [170,148,70, 6],
                [ 57, 42,23,15],
                [102, 94,34, 0],
                [ 66, 43,39,15]].toTensor()

    check: m1.astype(float) * m2.astype(float) == m1m2.astype(float)
    check: m1.astype(Complex[float64]) * m2.astype(Complex[float64]) == m1m2.astype(Complex[float64])

    # from http://www.calcul.com/show/calculator/matrix-multiplication?matrix1=[[%222%22,%224%22,%223%22,%221%22,%223%22,%221%22,%223%22,%221%22],[%221%22,%222%22,%221%22,%221%22,%222%22,%220%22,%224%22,%223%22],[%222%22,%220%22,%220%22,%223%22,%220%22,%224%22,%224%22,%221%22],[%221%22,%221%22,%224%22,%220%22,%223%22,%221%22,%223%22,%220%22],[%223%22,%224%22,%221%22,%221%22,%224%22,%222%22,%223%22,%224%22],[%222%22,%224%22,%220%22,%222%22,%223%22,%223%22,%223%22,%224%22],[%223%22,%220%22,%220%22,%223%22,%221%22,%224%22,%223%22,%221%22],[%224%22,%223%22,%222%22,%224%22,%221%22,%220%22,%220%22,%220%22]]&matrix2=[[%222%22,%222%22,%220%22,%224%22,%220%22,%220%22,%224%22,%222%22],[%222%22,%220%22,%220%22,%221%22,%221%22,%221%22,%223%22,%221%22],[%220%22,%222%22,%222%22,%220%22,%222%22,%222%22,%223%22,%223%22],[%220%22,%220%22,%221%22,%220%22,%224%22,%222%22,%224%22,%221%22],[%220%22,%220%22,%221%22,%223%22,%224%22,%222%22,%224%22,%222%22],[%224%22,%223%22,%224%22,%221%22,%224%22,%224%22,%220%22,%223%22],[%223%22,%223%22,%220%22,%222%22,%221%22,%222%22,%223%22,%223%22],[%222%22,%221%22,%222%22,%221%22,%222%22,%224%22,%224%22,%221%22]]&operator=*
    # (N x N) * (N x N) with N multiple of block size

    let n1 = [[2, 4,  3,  1,  3,  1,  3,  1],
              [1, 2,  1,  1,  2,  0,  4,  3],
              [2, 0,  0,  3,  0,  4,  4,  1],
              [1, 1,  4,  0,  3,  1,  3,  0],
              [3, 4,  1,  1,  4,  2,  3,  4],
              [2, 4,  0,  2,  3,  3,  3,  4],
              [3, 0,  0,  3,  1,  4,  3,  1],
              [4, 3,  2,  4,  1,  0,  0,  0]].toTensor()


    let n2 = [[2, 2,  0,  4,  0,  0,  4,  2],
              [2, 0,  0,  1,  1,  1,  3,  1],
              [0, 2,  2,  0,  2,  2,  3,  3],
              [0, 0,  1,  0,  4,  2,  4,  1],
              [0, 0,  1,  3,  4,  2,  4,  2],
              [4, 3,  4,  1,  4,  4,  0,  3],
              [3, 3,  0,  2,  1,  2,  3,  3],
              [2, 1,  2,  1,  2,  4,  4,  1]].toTensor()

    let n1n2 = [[27,23,16,29,35,32,58,37],
                [24,19,11,23,26,30,49,27],
                [34,29,21,21,34,34,36,32],
                [17,22,15,21,28,25,40,33],
                [39,27,23,40,45,46,72,41],
                [41,26,25,34,47,48,65,38],
                [33,28,22,26,37,34,41,33],
                [14,12, 9,22,27,17,51,23]].toTensor()

    check: n1.astype(float) * n2.astype(float) == n1n2.astype(float)
    check: n1.astype(Complex[float64]) * n2.astype(Complex[float64]) == n1n2.astype(Complex[float64])

  when compileOption("boundChecks") and not defined(openmp):
    test "GEMM - Bounds checking":
      let c = @[@[1'f32,2,3],@[4'f32,5,6]].toTensor()

      expect(IndexError):
        discard c * c
  else:
    echo "Bound-checking is disabled or OpenMP is used. The out-of-bounds checking test has been skipped."

  test "GEMV - General Matrix to Vector Multiplication":
    ## TODO: test with slices
    let d_int = @[@[1,-1,2],@[0,-3,1]].toTensor()
    let e_int = @[2, 1, 0].toTensor()
    let te_int = @[1, -3].toTensor()

    # GEMV integer fallback test - see dedicated section for extensive test
    check: d_int * e_int == te_int

    let d_float = @[@[1.0,-1,2],@[0.0,-3,1]].toTensor()
    let e_float = @[2.0, 1, 0].toTensor()

    check: d_float * e_float == te_int.map(x => x.float64)

  test "GEMM and GEMV with transposed matrices":
    let a = @[@[1.0,2,3],@[4.0,5,6]].toTensor()
    let b = @[@[7.0, 8],@[9.0, 10],@[11.0, 12]].toTensor()

    let at = @[@[1.0,4],@[2.0,5],@[3.0,6]].toTensor()

    let expected = @[@[58.0,64],@[139.0,154]].toTensor()

    check: transpose(at) * b == expected
    check: transpose(at.astype(Complex[float64])) * b.astype(Complex[float64]) == expected.astype(Complex[float64])

    let bt = @[@[7.0, 9, 11],@[8.0, 10, 12]].toTensor()

    check: a * transpose(bt) == expected

    check: transpose(at) * transpose(bt) == expected

    let d = @[@[1.0,-1,2],@[0.0,-3,1]].toTensor()
    let e = @[2.0, 1, 0].toTensor()

    let dt = @[@[1.0,0],@[-1.0,-3],@[2.0,1]].toTensor()

    check: d * e == transpose(dt) * e

  test "GEMM with sliced matrices in column-major order":
    # http://www.cs.otago.ac.nz/cosc453/student_tutorials/principal_components.pdf

    let a =  [[ 0.6899999999999999,  0.4900000000000002],
              [-1.31,               -1.21],
              [ 0.3900000000000001,  0.9900000000000002],
              [ 0.08999999999999986, 0.2900000000000005],
              [ 1.29,                1.09],
              [ 0.4899999999999998,  0.7900000000000005],
              [ 0.1899999999999999, -0.3099999999999996],
              [-0.8100000000000001, -0.8099999999999996],
              [-0.3100000000000001, -0.3099999999999996],
              [-0.71,               -1.01]].toTensor

    var eigvecs: Tensor[float64]

    tensorCpu(2, 2, eigvecs, colMajor)
    eigvecs.storage.Fdata = newSeq[float64](eigvecs.size)

    eigvecs[0, 0] = -0.735178655544408
    eigvecs[0, 1] = 0.6778733985280118
    eigvecs[1, 0] = 0.6778733985280118
    eigvecs[1, 1] = 0.735178655544408

    let val = a * eigvecs[_, ^1..0|-1]

    let expected = [[ 0.827970186, -0.175115307],
                    [-1.77758033,   0.142857227],
                    [ 0.992197494,  0.384374989],
                    [ 0.274210416,  0.130417207],
                    [ 1.67580142,  -0.209498461],
                    [ 0.912949103,  0.175282444],
                    [-0.0991094375,-0.349824698],
                    [-1.14457216,   0.0464172582],
                    [-0.438046137,  0.0177646297],
                    [-1.22382056,  -0.162675287]].toTensor

    check:
      expected.mean_absolute_error(val) < 1e-9

  test "Scalar/dot product":
    ## TODO: test with slices
    let u_int = @[1, 3, -5].toTensor()
    let v_int = @[4, -2, -1].toTensor()

    check: dot(u_int,v_int) == 3

    let u_float = @[1'f64, 3, -5].toTensor()
    let v_float = @[4'f64, -2, -1].toTensor()

    check: dot(u_float,v_float) == 3.0

  test "Multiplication/division by scalar":
    let u_int = @[1, 3, -5].toTensor()
    let u_expected = @[2, 6, -10].toTensor()

    check: 2 * u_int == u_expected
    check: u_int * 2 == u_expected

    let u_float = @[1'f64, 3, -5].toTensor()

    let ufl_expected = @[2'f64, 6, -10].toTensor()
    check: ufl_expected / 2 == u_float
    check: ufl_expected.astype(Complex[float64]) / complex(2'f64) == u_float.astype(Complex[float64])

  test "Multiplication/division by scalar (inplace)":
    var u_int = @[1, 3, -5].toTensor()
    let u_expected = @[2, 6, -10].toTensor()
    u_int *= 2
    check: u_int == u_expected

    var u_float = @[1'f64, 3, -5].toTensor()
    let ufl_expected = @[2'f64, 6, -10].toTensor()
    u_float *= 2.0'f64
    check: ufl_expected == u_float

    var u_complex = @[1'f64, 3, -5].toTensor()
    let ucl_expected = @[2'f64, 6, -10].toTensor()
    u_complex *= 2.0'f64
    check: ucl_expected == u_complex

    block:
      var u_int = @[1, 3, -6].toTensor()
      let u_expected = @[0, 1, -3].toTensor()
      u_int /= 2
      check: u_int == u_expected

      var u_float = @[1'f64, 3, -5].toTensor()
      let ufl_expected = @[0.5'f64, 1.5, -2.5].toTensor()
      u_float /= 2.0'f64
      check: ufl_expected == u_float

      var u_complex = @[1'f64, 3, -5].toTensor()
      let ucl_expected = @[0.5'f64, 1.5, -2.5].toTensor()
      u_complex /= 2.0'f64
      check: ucl_expected == u_complex

  test "Tensor addition and substraction":
    let u_int = @[1, 3, -5].toTensor()
    let v_int = @[1, 1, 1].toTensor()
    let expected_add = @[2, 4, -4].toTensor()
    let expected_sub = @[0, 2, -6].toTensor()

    check: u_int + v_int == expected_add
    check: u_int - v_int == expected_sub
    check: u_int.astype(Complex[float64]) + v_int.astype(Complex[float64]) == expected_add.astype(Complex[float64])
    check: u_int.astype(Complex[float64]) - v_int.astype(Complex[float64]) == expected_sub.astype(Complex[float64])

  test "Tensor addition and substraction (inplace)":
    var u_int = @[1, 3, -5].toTensor()
    let v_int = @[1, 1, 1].toTensor()
    let expected_add = @[2, 4, -4].toTensor()
    let expected_sub = @[0, 2, -6].toTensor()

    u_int += v_int
    check: u_int == expected_add

    u_int -= v_int
    u_int -= v_int
    check: u_int == expected_sub
    block:
      var u_complex = @[1, 3, -5].toTensor().astype(Complex[float64])
      let v_complex = @[1, 1, 1].toTensor().astype(Complex[float64])
      let expected_add = @[2, 4, -4].toTensor().astype(Complex[float64])
      let expected_sub = @[0, 2, -6].toTensor().astype(Complex[float64])

      u_complex += v_complex
      check: u_complex == expected_add

      u_complex -= v_complex
      u_complex -= v_complex
      check: u_complex == expected_sub

  test "Tensor negative":
    let u_int = @[-1, 0, 2].toTensor()
    let expected_add = @[1, 0, -2].toTensor()

    check: - u_int == expected_add

  test "Addition-Substraction - slices":
    let a = @[@[1.0,2,3],@[4.0,5,6], @[7.0,8,9]].toTensor()
    let a_t = a.transpose()
    let a_c = a.astype(Complex[float64])
    let a_tc = a_t.astype(Complex[float64])

    check: a[0..1, 0..1] + a_t[0..1, 0..1] == [[2.0, 6], [6.0, 10]].toTensor()
    check: a[1..2, 1..2] - a_t[1..2, 1..2] == [[0.0, -2], [2.0, 0]].toTensor()
    check: a_c[0..1, 0..1] + a_tc[0..1, 0..1] == [[2.0, 6], [6.0, 10]].toTensor().astype(Complex[float64])
    check: a_c[1..2, 1..2] - a_tc[1..2, 1..2] == [[0.0, -2], [2.0, 0]].toTensor().astype(Complex[float64])

  when compileOption("boundChecks") and not defined(openmp):
    # OpenMP backend is crashing when exceptions are thrown due to GC alloc.
    test "Addition-Substraction - Bounds checking":
      let a = [[1.0,2,3], [4.0,5,6], [7.0,8,9]].toTensor()
      let a_t = a.transpose()

      expect(ValueError):
        discard a[1..2,1..2] + a_t

      expect(ValueError):
        discard a - a_t[1..2,1..2]
  else:
    echo  "Bound-checking is disabled as OpenMP backend crashes when exception are thrown (due to GC alloc). " &
          "The addition bounds-check checking test has been skipped."

  test "Integer Matrix-Vector Multiplication fallback":
    let a = [[1,2,3],
             [4,5,6],
             [7,8,9]].toTensor()
    let x = [2,1,3].toTensor()

    let ax = [13,31,49].toTensor()

    check: a * x == ax

    # example from http://www.calcul.com/show/calculator/matrix-multiplication?matrix1=[[%22-87%22,%2244%22,%2213%22,%221%22],[%228%22,%22-16%22,%228%22,%2291%22],[%226%22,%22-2%22,%2256%22,%22-56%22],[%2282%22,%2270%22,%2234%22,%2223%22],[%2252%22,%22-70%22,%220%22,%2253%22],[%2235%22,%2294%22,%2239%22,%2236%22]]&matrix2=[[%22-91%22],[%2281%22],[%2269%22],[%22-75%22]]&operator=*

    let b = [[-87, 44, 13, 1],
             [8, -16, 8, 91],
             [6, -2, 56, -56],
             [82, 70, 34, 23],
             [52, -70, 0, 53],
             [35, 94, 39, 36]].toTensor()

    let u = [-91, 81, 69, -75].toTensor()

    let bu = [12303, -8297, 7356, -1171, -14377, 4420].toTensor()

    check: b * u == bu

    # Now we check with non-contiguous slices
    # [[53, -70]    *  [[69]
    #  [-56, -2]]   *   [-91]]
    # http://www.calcul.com/show/calculator/matrix-multiplication?matrix1=[[%2253%22,%22-70%22],[%22-56%22,%22-2%22]]&matrix2=[[%2269%22],[%2281%22]]&operator=*

    let b2 = b[4..2|-2, 3..1|-2]

    let u2 = u[2..0|-2]

    check: b2*u2 == [10027, -3682].toTensor()

    # Now with optimized C contiguous slices
    #  [[-56, -2]   *   [[69]
    #   [23, 70]]  *   [-91]]
    # http://www.calcul.com/show/calculator/matrix-multiplication?matrix1=[[%2253%22,%22-70%22],[%22-56%22,%22-2%22]]&matrix2=[[%2269%22],[%2281%22]]&operator=*



    let b3 = b[2..3, 3..1|-2]

    check: b3*u2 == [-3682, -4783].toTensor

  test "Integer Matrix-Matrix Multiplication fallback":

    ## TODO: test with slices
    let a = [[1,2,3],
             [4,5,6]].toTensor()

    let b = [[7,  8],
             [9, 10],
             [11,12]].toTensor()

    let ab = [[ 58, 64],
              [139,154]].toTensor()

    check: a * b == ab

    # example from http://www.intmath.com/matrices-determinants/matrix-multiplication-examples.php
    # (M x K) * (K x N) with M < N
    let u = [[-2,-3,-1],
             [ 3, 0, 4]].toTensor()
    let v = [[ 1, 5, 2,-1],
             [-3, 0, 3, 4],
             [ 6,-2, 7,-4]].toTensor()

    let uv = [[ 1,-8,-20, -6],
              [27, 7, 34,-19]].toTensor()

    check: u * v == uv

    # from http://www.calcul.com/show/calculator/matrix-multiplication_;5;5;5;5?matrix1=[[%225%22,%226%22,%225%22,%228%22],[%228%22,%222%22,%228%22,%228%22],[%220%22,%225%22,%224%22,%220%22],[%224%22,%220%22,%225%22,%226%22],[%224%22,%225%22,%220%22,%223%22]]&matrix2=[[%225%22,%223%22,%226%22,%220%22],[%225%22,%222%22,%223%22,%223%22],[%228%22,%228%22,%222%22,%220%22],[%227%22,%227%22,%220%22,%220%22]]&operator=*
    # (M x K) * (K x N) with M > N and M > block-size (4x4)
    let m1 = [[5,6,5,8],
              [8,2,8,8],
              [0,5,4,0],
              [4,0,5,6],
              [4,5,0,3]].toTensor()
    let m2 = [[5,3,6,0],
              [5,2,3,3],
              [8,8,2,0],
              [7,7,0,0]].toTensor()

    let m1m2 = [[151,123,58,18],
                [170,148,70, 6],
                [ 57, 42,23,15],
                [102, 94,34, 0],
                [ 66, 43,39,15]].toTensor()

    check: m1 * m2 == m1m2

    # from http://www.calcul.com/show/calculator/matrix-multiplication?matrix1=[[%222%22,%224%22,%223%22,%221%22,%223%22,%221%22,%223%22,%221%22],[%221%22,%222%22,%221%22,%221%22,%222%22,%220%22,%224%22,%223%22],[%222%22,%220%22,%220%22,%223%22,%220%22,%224%22,%224%22,%221%22],[%221%22,%221%22,%224%22,%220%22,%223%22,%221%22,%223%22,%220%22],[%223%22,%224%22,%221%22,%221%22,%224%22,%222%22,%223%22,%224%22],[%222%22,%224%22,%220%22,%222%22,%223%22,%223%22,%223%22,%224%22],[%223%22,%220%22,%220%22,%223%22,%221%22,%224%22,%223%22,%221%22],[%224%22,%223%22,%222%22,%224%22,%221%22,%220%22,%220%22,%220%22]]&matrix2=[[%222%22,%222%22,%220%22,%224%22,%220%22,%220%22,%224%22,%222%22],[%222%22,%220%22,%220%22,%221%22,%221%22,%221%22,%223%22,%221%22],[%220%22,%222%22,%222%22,%220%22,%222%22,%222%22,%223%22,%223%22],[%220%22,%220%22,%221%22,%220%22,%224%22,%222%22,%224%22,%221%22],[%220%22,%220%22,%221%22,%223%22,%224%22,%222%22,%224%22,%222%22],[%224%22,%223%22,%224%22,%221%22,%224%22,%224%22,%220%22,%223%22],[%223%22,%223%22,%220%22,%222%22,%221%22,%222%22,%223%22,%223%22],[%222%22,%221%22,%222%22,%221%22,%222%22,%224%22,%224%22,%221%22]]&operator=*
    # (N x N) * (N x N) with N multiple of block size

    let n1 = [[2, 4,  3,  1,  3,  1,  3,  1],
              [1, 2,  1,  1,  2,  0,  4,  3],
              [2, 0,  0,  3,  0,  4,  4,  1],
              [1, 1,  4,  0,  3,  1,  3,  0],
              [3, 4,  1,  1,  4,  2,  3,  4],
              [2, 4,  0,  2,  3,  3,  3,  4],
              [3, 0,  0,  3,  1,  4,  3,  1],
              [4, 3,  2,  4,  1,  0,  0,  0]].toTensor()


    let n2 = [[2, 2,  0,  4,  0,  0,  4,  2],
              [2, 0,  0,  1,  1,  1,  3,  1],
              [0, 2,  2,  0,  2,  2,  3,  3],
              [0, 0,  1,  0,  4,  2,  4,  1],
              [0, 0,  1,  3,  4,  2,  4,  2],
              [4, 3,  4,  1,  4,  4,  0,  3],
              [3, 3,  0,  2,  1,  2,  3,  3],
              [2, 1,  2,  1,  2,  4,  4,  1]].toTensor()

    let n1n2 = [[27,23,16,29,35,32,58,37],
                [24,19,11,23,26,30,49,27],
                [34,29,21,21,34,34,36,32],
                [17,22,15,21,28,25,40,33],
                [39,27,23,40,45,46,72,41],
                [41,26,25,34,47,48,65,38],
                [33,28,22,26,37,34,41,33],
                [14,12, 9,22,27,17,51,23]].toTensor()

    check: n1 * n2 == n1n2

  test "complex matrix product":
    # [[1. + 1.j, 2. + 2.j, 3. + 3.j]    [[1. + 1.j, 4. + 4.j],     [[0. + 28.j, 0. + 64.j],
    #  [4. +4.j, 5. + 5.j, 6. + 6.j]] *  [2. + 2.j, 5. +5.j],  ==  [0. + 64.j, 0. + 154.j]
    #                               [3. + 3.j, 6. + 6.j]]
    let m1 = [[1,2,3],[4,5,6]].toTensor().astype(Complex[float64])
    let m2 = m1 * complex64(0,1)
    let m3 = m1+m2
    let m4 = m3.transpose()
    let m5 = m3 * m4
    let m6 = [[28,64],[64,154]].toTensor().astype(Complex[float64])
    check: m5 == m6 * complex64(0,1)
