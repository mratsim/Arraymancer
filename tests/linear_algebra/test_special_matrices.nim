# Copyright (c) 2018 the Arraymancer contributors
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).

import ../../src/arraymancer
import unittest, math
import std/strformat

proc main() =
  suite "Diagonals":
    test "Identity and Eye":
      block:
        let identity = identity[float64](3)
        let expected_identity = [[1.0, 0.0, 0.0],
                                 [0.0, 1.0, 0.0],
                                 [0.0, 0.0, 1.0]].toTensor
        check:
          identity == expected_identity

        let eye = eye[float64](3, 4)
        let expected_eye = [[1.0, 0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0, 0.0]].toTensor
        check:
          eye == expected_eye

    test "Diagonal Matrix creation (diag)":
      block:
        # Issue: should implement anti?
        let square_diag = diag([1, 2, 3].toTensor)
        let expected_square_diag = [[1, 0, 0],
                                    [0, 2, 0],
                                    [0, 0, 3]].toTensor
        let upper1_diag = diag([1, 2].toTensor, 1)
        let expected_upper1_diag = [[0, 1, 0],
                                    [0, 0, 2],
                                    [0, 0, 0]].toTensor
        let lower1_diag = diag([1, 2].toTensor, -1)
        let expected_lower1_diag = [[0, 0, 0],
                                   [1, 0, 0],
                                   [0, 2, 0]].toTensor
        check:
          square_diag == expected_square_diag
          upper1_diag == expected_upper1_diag
          lower1_diag == expected_lower1_diag

    test "Get diagonal":
      block:
        let a = arange(4*5).reshape(4, 5)
        let expected_upper2_diagonal = @[2, 8, 14].toTensor
        let expected_upper1_diagonal = @[1, 7, 13, 19].toTensor
        let expected_center_diagonal = @[0, 6, 12, 18].toTensor
        let expected_lower2_diagonal = @[10, 16].toTensor

        check:
          a.diagonal(2) == expected_upper2_diagonal
          a.diagonal(1) == expected_upper1_diagonal
          a.diagonal() == expected_center_diagonal
          a.diagonal(-2) == expected_lower2_diagonal

      block:
        let a = arange(4*5).reshape(4, 5)
        let expected_upper2_antidiagonal = @[5, 1].toTensor
        let expected_center_antidiagonal = @[15, 11, 7, 3].toTensor
        let expected_lower1_antidiagonal = @[16, 12, 8, 4].toTensor
        let expected_lower2_antidiagonal = @[17, 13, 9].toTensor

        check:
          a.diagonal(2, anti=true) == expected_upper2_antidiagonal
          a.diagonal(anti=true) == expected_center_antidiagonal
          a.diagonal(-1, anti=true) == expected_lower1_antidiagonal
          a.diagonal(-2, anti=true) == expected_lower2_antidiagonal

    test "Set diagonal":
      block:
        var a = zeros[int](3, 4).with_diagonal([10, 20, 30].toTensor, 1)
        a.set_diagonal([100, 200].toTensor, 2)
        a.set_diagonal([1, 2, 3].toTensor)
        a.set_diagonal([-10, -20].toTensor, -1)
        let expected_a = [[  1,  10, 100,   0],
                          [-10,   2,  20, 200],
                          [  0, -20,   3,  30]].toTensor
        check:
          a == expected_a

      block: # Check the "anti" flag
        var a = zeros[int](4,3).with_diagonal([1, 2, 3].toTensor, anti=true)
        a.set_diagonal([10, 20, 30].toTensor, 1, anti=true)
        a.set_diagonal([100, 200].toTensor, 2, anti=true)
        a.set_diagonal([-10, -20].toTensor, -1, anti=true)
        let expected_a = [[  0, 200,  30],
                          [100,  20,   3],
                          [ 10,   2, -20],
                          [  1, -10,   0]].toTensor
        check:
          a == expected_a

    test "Triangular Matrices":
      block:
        let expected_t1 = [[1.0, 0.0, 0.0],
                           [1.0, 1.0, 0.0],
                           [1.0, 1.0, 1.0]].toTensor
        let expected_t2 = [[1.0, 1.0, 1.0],
                           [0.0, 1.0, 1.0],
                           [0.0, 0.0, 1.0]].toTensor
        let expected_t3 = [[0.0, 1.0, 1.0],
                           [0.0, 0.0, 1.0],
                           [0.0, 0.0, 0.0]].toTensor
        let t1 = tri[float](3, 3)
        let t2 = tri[float](3, 3, upper = true)
        let t3 = tri[float](3, 3, k=1, upper = true)

        let t1b = tri[float](t1.shape)
        let t2b = tri[float](t1.shape, upper = true)
        let t3b = tri[float](t1.shape, k = 1, upper = true)

        check:
          t1 == expected_t1
          t2 == expected_t2
          t3 == expected_t3
          t1b == expected_t1
          t2b == expected_t2
          t3b == expected_t3

  suite "meshgrid":
    test "meshgrid-2D":
      block:
        let expected_xy_x = [[0, 1],
                             [0, 1],
                             [0, 1]].toTensor
        let expected_xy_y = [[2, 2],
                             [3, 3],
                             [4, 4]].toTensor
        let expected_ij_x = [[0, 0, 0],
                             [1, 1, 1]].toTensor
        let expected_ij_y = [[2, 3, 4],
                             [2, 3, 4]].toTensor
        let xy = meshgrid(arange(2), arange(2, 5), indexing=xygrid)
        let ij = meshgrid(arange(2), arange(2, 5), indexing=ijgrid)

        check:
          xy == @[expected_xy_x, expected_xy_y]
          ij == @[expected_ij_x, expected_ij_y]

    test "meshgrid-3D":
      block:
        let expected_xy_x = [[[0, 0, 0, 0],
                             [1, 1, 1, 1]],
                            [[0, 0, 0, 0],
                             [1, 1, 1, 1]],
                            [[0, 0, 0, 0],
                             [1, 1, 1, 1]]].toTensor
        let expected_xy_y = [[[2, 2, 2, 2],
                              [2, 2, 2, 2]],
                             [[3, 3, 3, 3],
                              [3, 3, 3, 3]],
                             [[4, 4, 4, 4],
                              [4, 4, 4, 4]]].toTensor
        let expected_xy_z = [[[5, 6, 7, 8],
                              [5, 6, 7, 8]],
                             [[5, 6, 7, 8],
                              [5, 6, 7, 8]],
                             [[5, 6, 7, 8],
                              [5, 6, 7, 8]]].toTensor
        let expected_ij_x = [[[0, 0, 0, 0],
                              [0, 0, 0, 0],
                              [0, 0, 0, 0]],
                             [[1, 1, 1, 1],
                              [1, 1, 1, 1],
                              [1, 1, 1, 1]]].toTensor
        let expected_ij_y = [[[2, 2, 2, 2],
                              [3, 3, 3, 3],
                              [4, 4, 4, 4]],
                             [[2, 2, 2, 2],
                              [3, 3, 3, 3],
                              [4, 4, 4, 4]]].toTensor
        let expected_ij_z = [[[5, 6, 7, 8],
                              [5, 6, 7, 8],
                              [5, 6, 7, 8]],
                             [[5, 6, 7, 8],
                              [5, 6, 7, 8],
                              [5, 6, 7, 8]]].toTensor
        let xy = meshgrid(arange(2), arange(2, 5), arange(5, 9), indexing=xygrid)
        let ij = meshgrid(arange(2), arange(2, 5), arange(5, 9), indexing=ijgrid)

        check:
          xy == @[expected_xy_x, expected_xy_y, expected_xy_z]
          ij == @[expected_ij_x, expected_ij_y, expected_ij_z]


main()
GC_fullCollect()
