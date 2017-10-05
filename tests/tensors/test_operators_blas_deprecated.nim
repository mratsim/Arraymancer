# Copyright 2017 Mamy André-Ratsimbazafy
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
import unittest, future

suite "BLAS (Basic Linear Algebra Subprograms)":
  test "GEMM - General Matrix to Matrix Multiplication":
    ## TODO: test with slices
    let a = [[1.0,2,3],
             [4.0,5,6]].toTensor(Cpu)

    let b = [[7.0,  8],
             [9.0, 10],
             [11.0,12]].toTensor(Cpu)

    let ab = [[ 58.0, 64],
              [139.0,154]].toTensor(Cpu)

    check: a * b == ab

    # example from http://www.intmath.com/matrices-determinants/matrix-multiplication-examples.php
    # (M x K) * (K x N) with M < N
    let u = [[-2,-3,-1],
             [ 3, 0, 4]].toTensor(Cpu)
    let v = [[ 1, 5, 2,-1],
             [-3, 0, 3, 4],
             [ 6,-2, 7,-4]].toTensor(Cpu)

    let uv = [[ 1,-8,-20, -6],
              [27, 7, 34,-19]].toTensor(Cpu)

    check: u.astype(float32) * v.astype(float32) == uv.astype(float32)

    # from http://www.calcul.com/show/calculator/matrix-multiplication_;5;5;5;5?matrix1=[[%225%22,%226%22,%225%22,%228%22],[%228%22,%222%22,%228%22,%228%22],[%220%22,%225%22,%224%22,%220%22],[%224%22,%220%22,%225%22,%226%22],[%224%22,%225%22,%220%22,%223%22]]&matrix2=[[%225%22,%223%22,%226%22,%220%22],[%225%22,%222%22,%223%22,%223%22],[%228%22,%228%22,%222%22,%220%22],[%227%22,%227%22,%220%22,%220%22]]&operator=*
    # (M x K) * (K x N) with M > N and M > block-size (4x4)
    let m1 = [[5,6,5,8],
              [8,2,8,8],
              [0,5,4,0],
              [4,0,5,6],
              [4,5,0,3]].toTensor(Cpu)
    let m2 = [[5,3,6,0],
              [5,2,3,3],
              [8,8,2,0],
              [7,7,0,0]].toTensor(Cpu)

    let m1m2 = [[151,123,58,18],
                [170,148,70, 6],
                [ 57, 42,23,15],
                [102, 94,34, 0],
                [ 66, 43,39,15]].toTensor(Cpu)
    
    check: m1.astype(float) * m2.astype(float) == m1m2.astype(float)

    # from http://www.calcul.com/show/calculator/matrix-multiplication?matrix1=[[%222%22,%224%22,%223%22,%221%22,%223%22,%221%22,%223%22,%221%22],[%221%22,%222%22,%221%22,%221%22,%222%22,%220%22,%224%22,%223%22],[%222%22,%220%22,%220%22,%223%22,%220%22,%224%22,%224%22,%221%22],[%221%22,%221%22,%224%22,%220%22,%223%22,%221%22,%223%22,%220%22],[%223%22,%224%22,%221%22,%221%22,%224%22,%222%22,%223%22,%224%22],[%222%22,%224%22,%220%22,%222%22,%223%22,%223%22,%223%22,%224%22],[%223%22,%220%22,%220%22,%223%22,%221%22,%224%22,%223%22,%221%22],[%224%22,%223%22,%222%22,%224%22,%221%22,%220%22,%220%22,%220%22]]&matrix2=[[%222%22,%222%22,%220%22,%224%22,%220%22,%220%22,%224%22,%222%22],[%222%22,%220%22,%220%22,%221%22,%221%22,%221%22,%223%22,%221%22],[%220%22,%222%22,%222%22,%220%22,%222%22,%222%22,%223%22,%223%22],[%220%22,%220%22,%221%22,%220%22,%224%22,%222%22,%224%22,%221%22],[%220%22,%220%22,%221%22,%223%22,%224%22,%222%22,%224%22,%222%22],[%224%22,%223%22,%224%22,%221%22,%224%22,%224%22,%220%22,%223%22],[%223%22,%223%22,%220%22,%222%22,%221%22,%222%22,%223%22,%223%22],[%222%22,%221%22,%222%22,%221%22,%222%22,%224%22,%224%22,%221%22]]&operator=*
    # (N x N) * (N x N) with N multiple of block size

    let n1 = [[2, 4,  3,  1,  3,  1,  3,  1],
              [1, 2,  1,  1,  2,  0,  4,  3],
              [2, 0,  0,  3,  0,  4,  4,  1],
              [1, 1,  4,  0,  3,  1,  3,  0],
              [3, 4,  1,  1,  4,  2,  3,  4],
              [2, 4,  0,  2,  3,  3,  3,  4],
              [3, 0,  0,  3,  1,  4,  3,  1],
              [4, 3,  2,  4,  1,  0,  0,  0]].toTensor(Cpu)


    let n2 = [[2, 2,  0,  4,  0,  0,  4,  2],
              [2, 0,  0,  1,  1,  1,  3,  1],
              [0, 2,  2,  0,  2,  2,  3,  3],
              [0, 0,  1,  0,  4,  2,  4,  1],
              [0, 0,  1,  3,  4,  2,  4,  2],
              [4, 3,  4,  1,  4,  4,  0,  3],
              [3, 3,  0,  2,  1,  2,  3,  3],
              [2, 1,  2,  1,  2,  4,  4,  1]].toTensor(Cpu)
    
    let n1n2 = [[27,23,16,29,35,32,58,37],
                [24,19,11,23,26,30,49,27],
                [34,29,21,21,34,34,36,32],
                [17,22,15,21,28,25,40,33],
                [39,27,23,40,45,46,72,41],
                [41,26,25,34,47,48,65,38],
                [33,28,22,26,37,34,41,33],
                [14,12, 9,22,27,17,51,23]].toTensor(Cpu)

    check: n1.astype(float) * n2.astype(float) == n1n2.astype(float)

  test "GEMM - Bounds checking":
    let c = @[@[1'f32,2,3],@[4'f32,5,6]]
    let tc = c.toTensor(Cpu)

    when compiles(tc * tb): check: false
    expect(IndexError):
       discard tc * tc

  test "GEMV - General Matrix to Vector Multiplication":
    ## TODO: test with slices
    let d_int = @[@[1,-1,2],@[0,-3,1]]
    let e_int = @[2, 1, 0]
    let tde_expected_int = @[1, -3]

    let td_int = d_int.toTensor(Cpu)
    let te_int = e_int.toTensor(Cpu)

    ## TODO integer fallback
    # check: td_int * te_int == tde_expected_int.toTensor(Cpu)

    let d_float = @[@[1.0,-1,2],@[0.0,-3,1]]
    let e_float = @[2.0, 1, 0]

    let td_float = d_float.toTensor(Cpu)
    let te_float = e_float.toTensor(Cpu)

    check: td_float * te_float == tde_expected_int.toTensor(Cpu).fmap(x => x.float64)

  test "GEMM and GEMV with transposed matrices":
    let a = @[@[1.0,2,3],@[4.0,5,6]]
    let ta = a.toTensor(Cpu)
    let b = @[@[7.0, 8],@[9.0, 10],@[11.0, 12]]
    let tb = b.toTensor(Cpu)


    let at = @[@[1.0,4],@[2.0,5],@[3.0,6]]
    let tat = at.toTensor(Cpu)

    let expected = @[@[58.0,64],@[139.0,154]]
    let t_expected = expected.toTensor(Cpu)

    check: transpose(tat) * tb == t_expected

    let bt = @[@[7.0, 9, 11],@[8.0, 10, 12]]
    let tbt = bt.toTensor(Cpu)

    check: ta * transpose(tbt) == t_expected

    check: transpose(tat) * transpose(tbt) == t_expected


    let d = @[@[1.0,-1,2],@[0.0,-3,1]]
    let e = @[2.0, 1, 0]

    let td = d.toTensor(Cpu)
    let te = e.toTensor(Cpu)

    let dt = @[@[1.0,0],@[-1.0,-3],@[2.0,1]]
    let tdt = dt.toTensor(Cpu)

    check: td * te == transpose(tdt) * te

  test "Scalar/dot product":
    ## TODO: test with slices
    let u_int = @[1, 3, -5]
    let v_int = @[4, -2, -1]

    let tu_int = u_int.toTensor(Cpu)
    let tv_int = v_int.toTensor(Cpu)

    check: dot(tu_int,tv_int) == 3


    let u_float = @[1'f64, 3, -5]
    let v_float = @[4'f64, -2, -1]

    let tu_float = u_float.toTensor(Cpu)
    let tv_float = v_float.toTensor(Cpu)

    check: dot(tu_float,tv_float) == 3.0

  test "Multiplication/division by scalar":
    let u_int = @[1, 3, -5]
    let u_expected = @[2, 6, -10]
    let tu_int = u_int.toTensor(Cpu)

    check: 2 * tu_int == u_expected.toTensor(Cpu)
    check: tu_int * 2 == u_expected.toTensor(Cpu)

    let u_float = @[1'f64, 3, -5]
    let tu_float = u_float.toTensor(Cpu)

    let ufl_expected = @[2'f64, 6, -10]
    check: ufl_expected.toTensor(Cpu) / 2 == tu_float

  test "Tensor addition and substraction":
    let u_int = @[1, 3, -5]
    let v_int = @[1, 1, 1]
    let expected_add = @[2, 4, -4]
    let expected_sub = @[0, 2, -6]
    let tu_int = u_int.toTensor(Cpu)
    let tv_int = v_int.toTensor(Cpu)

    check: tu_int + tv_int == expected_add.toTensor(Cpu)
    check: tu_int - tv_int == expected_sub.toTensor(Cpu)

  test "Addition-Substraction - slices":
    let a = @[@[1.0,2,3],@[4.0,5,6], @[7.0,8,9]]
    let ta = a.toTensor(Cpu)
    let ta_t = ta.transpose()

    check: ta[0..1, 0..1] + ta_t[0..1, 0..1] == [[2.0, 6], [6.0, 10]].toTensor(Cpu)
    check: ta[1..2, 1..2] - ta_t[1..2, 1..2] == [[0.0, -2], [2.0, 0]].toTensor(Cpu)

  test "Addition-Substraction - Bounds checking":
    let a = [[1.0,2,3], [4.0,5,6], [7.0,8,9]]
    let ta = a.toTensor(Cpu)
    let ta_t = ta.transpose()

    expect(ValueError):
      discard ta[1..2,1..2] + ta_t

    expect(ValueError):
      discard ta - ta_t[1..2,1..2]

  test "Integer Matrix-Vector Multiplication fallback":
    let a = [[1,2,3],
             [4,5,6],
             [7,8,9]].toTensor(Cpu)
    let x = [2,1,3].toTensor(Cpu)

    let ax = [13,31,49].toTensor(Cpu)

    check: a * x == ax

    # example from http://www.calcul.com/show/calculator/matrix-multiplication?matrix1=[[%22-87%22,%2244%22,%2213%22,%221%22],[%228%22,%22-16%22,%228%22,%2291%22],[%226%22,%22-2%22,%2256%22,%22-56%22],[%2282%22,%2270%22,%2234%22,%2223%22],[%2252%22,%22-70%22,%220%22,%2253%22],[%2235%22,%2294%22,%2239%22,%2236%22]]&matrix2=[[%22-91%22],[%2281%22],[%2269%22],[%22-75%22]]&operator=*

    let b = [[-87, 44, 13, 1],
             [8, -16, 8, 91],
             [6, -2, 56, -56],
             [82, 70, 34, 23],
             [52, -70, 0, 53],
             [35, 94, 39, 36]].toTensor(Cpu)
    
    let u = [-91, 81, 69, -75].toTensor(Cpu)

    let bu = [12303, -8297, 7356, -1171, -14377, 4420].toTensor(Cpu)

    check: b * u == bu

  test "Integer Matrix-Matrix Multiplication fallback":
    ## TODO: test with slices
    let a = [[1,2,3],
             [4,5,6]].toTensor(Cpu)

    let b = [[7,  8],
             [9, 10],
             [11,12]].toTensor(Cpu)

    let ab = [[ 58, 64],
              [139,154]].toTensor(Cpu)

    check: a * b == ab

    # example from http://www.intmath.com/matrices-determinants/matrix-multiplication-examples.php
    # (M x K) * (K x N) with M < N
    let u = [[-2,-3,-1],
             [ 3, 0, 4]].toTensor(Cpu)
    let v = [[ 1, 5, 2,-1],
             [-3, 0, 3, 4],
             [ 6,-2, 7,-4]].toTensor(Cpu)

    let uv = [[ 1,-8,-20, -6],
              [27, 7, 34,-19]].toTensor(Cpu)

    check: u * v == uv

    # from http://www.calcul.com/show/calculator/matrix-multiplication_;5;5;5;5?matrix1=[[%225%22,%226%22,%225%22,%228%22],[%228%22,%222%22,%228%22,%228%22],[%220%22,%225%22,%224%22,%220%22],[%224%22,%220%22,%225%22,%226%22],[%224%22,%225%22,%220%22,%223%22]]&matrix2=[[%225%22,%223%22,%226%22,%220%22],[%225%22,%222%22,%223%22,%223%22],[%228%22,%228%22,%222%22,%220%22],[%227%22,%227%22,%220%22,%220%22]]&operator=*
    # (M x K) * (K x N) with M > N and M > block-size (4x4)
    let m1 = [[5,6,5,8],
              [8,2,8,8],
              [0,5,4,0],
              [4,0,5,6],
              [4,5,0,3]].toTensor(Cpu)
    let m2 = [[5,3,6,0],
              [5,2,3,3],
              [8,8,2,0],
              [7,7,0,0]].toTensor(Cpu)

    let m1m2 = [[151,123,58,18],
                [170,148,70, 6],
                [ 57, 42,23,15],
                [102, 94,34, 0],
                [ 66, 43,39,15]].toTensor(Cpu)
    
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
              [4, 3,  2,  4,  1,  0,  0,  0]].toTensor(Cpu)


    let n2 = [[2, 2,  0,  4,  0,  0,  4,  2],
              [2, 0,  0,  1,  1,  1,  3,  1],
              [0, 2,  2,  0,  2,  2,  3,  3],
              [0, 0,  1,  0,  4,  2,  4,  1],
              [0, 0,  1,  3,  4,  2,  4,  2],
              [4, 3,  4,  1,  4,  4,  0,  3],
              [3, 3,  0,  2,  1,  2,  3,  3],
              [2, 1,  2,  1,  2,  4,  4,  1]].toTensor(Cpu)
    
    let n1n2 = [[27,23,16,29,35,32,58,37],
                [24,19,11,23,26,30,49,27],
                [34,29,21,21,34,34,36,32],
                [17,22,15,21,28,25,40,33],
                [39,27,23,40,45,46,72,41],
                [41,26,25,34,47,48,65,38],
                [33,28,22,26,37,34,41,33],
                [14,12, 9,22,27,17,51,23]].toTensor(Cpu)

    check: n1 * n2 == n1n2

