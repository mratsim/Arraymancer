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
import unittest, math
import complex except Complex64, Complex32

proc main() =
  suite "[Core] Testing aggregation functions":
    let t = [[0, 1, 2],
            [3, 4, 5],
            [6, 7, 8],
            [9, 10, 11]].toTensor()
    let t_c = t.asType(Complex[float64])

    test "Sum":
      check: t.sum == 66
      check: t_c.sum == complex(66'f64)
      let row_sum = [[18, 22, 26]].toTensor()
      let col_sum = [[3],
                    [12],
                    [21],
                    [30]].toTensor()
      check: t.sum(axis=0) == row_sum
      check: t.sum(axis=1) == col_sum
      check: t_c.sum(axis=0) == row_sum.asType(Complex[float64])
      check: t_c.sum(axis=1) == col_sum.asType(Complex[float64])

    test "Mean":
      check: t.asType(float).mean == 5.5 # Note: may fail due to float rounding
      check: t_c.mean == complex(5.5) # Note: may fail due to float rounding

      let row_mean = [[4.5, 5.5, 6.5]].toTensor()
      let col_mean = [[1.0],
                      [4.0],
                      [7.0],
                      [10.0]].toTensor()
      check: t.asType(float).mean(axis=0) == row_mean
      check: t.asType(float).mean(axis=1) == col_mean
      check: t_c.mean(axis=0) == row_mean.asType(Complex[float64])
      check: t_c.mean(axis=1) == col_mean.asType(Complex[float64])

    test "Product":
      let a = [[1,2,4],[8,16,32]].toTensor()
      let a_c = a.asType(Complex[float64])
      check: t.product() == 0
      check: a.product() == 32768
      check: a.asType(float).product() == 32768.0
      check: a.product(0) == [[8,32,128]].toTensor()
      check: a.product(1) == [[8],[4096]].toTensor()
      check: t_c.product() == complex(0.0)
      check: a_c.product() == complex(32768.0)
      check: a_c.product(0) == [[8,32,128]].toTensor().asType(Complex[float64])
      check: a_c.product(1) == [[8],[4096]].toTensor().asType(Complex[float64])

    test "Min":
      let a = [2,-1,3,-3,5,0].toTensor()
      check: a.min() == -3
      check: a.asType(float32).min() == -3.0f

      let b = [[1,2,3,-4],[0,4,-2,5]].toTensor()
      check: b.min(0) == [[0,2,-2,-4]].toTensor()
      check: b.min(1) == [[-4],[-2]].toTensor()
      check: b.asType(float32).min(0) == [[0.0f,2,-2,-4]].toTensor()
      check: b.asType(float32).min(1) == [[-4.0f],[-2.0f]].toTensor()

    test "Max":
      let a = [2,-1,3,-3,5,0].toTensor()
      check: a.max() == 5
      check: a.asType(float32).max() == 5.0f

      let b = [[1,2,3,-4],[0,4,-2,5]].toTensor()
      check: b.max(0) == [[1,4,3,5]].toTensor()
      check: b.max(1) == [[3],[5]].toTensor()
      check: b.asType(float32).max(0) == [[1.0f,4,3,5]].toTensor()
      check: b.asType(float32).max(1) == [[3.0f],[5.0f]].toTensor()

    test "Variance":
      let a = [-3.0,-2,-1,0,1,2,3].toTensor()
      check: abs(a.variance() - 4.6666666666667) < 1e-8
      let b = [[1.0,2,3,-4],[0.0,4,-2,5]].toTensor()
      check: b.variance(0) == [[0.5,2.0,12.5,40.5]].toTensor()
      check: (
        b.variance(1) -
        [[9.666666666666666], [10.91666666666667]].toTensor()
      ).abs().sum() < 1e-8

    test "Standard Deviation":
      let a = [-3.0,-2,-1,0,1,2,3].toTensor()
      check: abs(a.std() - 2.1602468994693) < 1e-8
      let b = [[1.0,2,3,-4],[0.0,4,-2,5]].toTensor()
      check: abs(
        b.std(0) -
        [[0.7071067811865476,1.414213562373095,
          3.535533905932738,6.363961030678928]].toTensor()
      ).abs().sum() < 1e-8
      check: abs(
        b.std(1) -
        [[3.109126351029605],[3.304037933599835]].toTensor()
      ).abs().sum() < 1e-8

    test "Argmax":
      let a =  [[0, 4, 7],
                [1, 9, 5],
                [3, 4, 1]].toTensor
      check: argmax(a, 0) == [[2, 1, 0]].toTensor
      check: argmax(a, 1) == [[2],
                              [1],
                              [1]].toTensor

      block:
        let a =  [[0, 1, 2],
                  [3, 4, 5]].toTensor
        check: argmax(a, 0) == [[1, 1, 1]].toTensor
        check: argmax(a, 1) == [[2],
                                [2]].toTensor

    test "Argmax_3D":
      let a = [
        [[1, 10, 5, 5, 7, 3], [8, 3, 7, 9, 3, 8], [5, 3, 7, 1, 4, 5], [8, 10, 5, 8, 9, 1], [10, 5, 2, 1, 5, 8]],
        [[10, 0, 1, 9, 0, 4], [5, 7, 10, 0, 7, 5], [6, 1, 1, 10, 2, 2], [6, 10, 1, 9, 7, 8], [10, 7, 5, 9, 1, 3]],
        [[9, 1, 2, 1, 5, 10], [6, 1, 7, 9, 3, 0], [2, 1, 4, 8, 5, 7], [5, 7, 0, 4, 3, 2], [2, 7, 5, 8, 5, 6]],
        [[2, 8, 5, 9, 1, 5], [5, 10, 6, 8, 0, 1], [0, 10, 0, 8, 6, 7], [5, 1, 4, 9, 3, 0], [1, 1, 4, 3, 9, 4]]
      ].toTensor

      check: argmax(a, 0) == [
        [[1, 0, 0, 1, 0, 2], [0, 3, 1, 0, 1, 0], [1, 3, 0, 1, 3, 2], [0, 0, 0, 1, 0, 1], [0, 1, 1, 1, 3, 0]]
      ].toTensor

      check: argmax(a, 1) == [
        [[4, 0, 1, 1, 3, 1]], [[0, 3, 1, 2, 1, 3]], [[0, 3, 1, 1, 0, 0]], [[1, 1, 1, 0, 4, 2]]
      ].toTensor

      check: argmax(a, 2) == [
        [[1], [3], [2], [1], [0]], [[0], [2], [3], [1], [0]], [[5], [3], [3], [1], [3]], [[3], [1], [1], [3], [4]]
      ].toTensor

    block:
      let a = [
        [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]],
        [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]
      ].toTensor

      check: argmax(a, 0) == [
        [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]
      ].toTensor

      check: argmax(a, 1) == [
        [[2, 2, 2, 2]], [[2, 2, 2, 2]]
      ].toTensor

      check: argmax(a, 2) == [
        [[3], [3], [3]], [[3], [3], [3]]
      ].toTensor

    test "Argmin":
      let a =  [[0, 4, 7],
                [1, 9, 5],
                [3, 4, 1]].toTensor
      check: argmin(a, 0) == [[0, 0, 2]].toTensor
      check: argmin(a, 1) == [[0],
                              [0],
                              [2]].toTensor

      block:
        let a =  [[0, 1, 2],
                  [3, 4, 5]].toTensor
        check: argmin(a, 0) == [[0, 0, 0]].toTensor
        check: argmin(a, 1) == [[0],
                                [0]].toTensor

    test "Argmin_3D":
      let a = [
        [[1 , 10, 5, 5, 7, 3 ], [8, 3 , 7 , 9, 3, 8], [5, 3 , 7, 1 , 4, 5], [8, 10, 5, 8, 9, 1], [10, 5, 2, 1, 5, 8]],
        [[10, 0 , 1, 9, 0, 4 ], [5, 7 , 10, 0, 7, 5], [6, 1 , 1, 10, 2, 2], [6, 10, 1, 9, 7, 8], [10, 7, 5, 9, 1, 3]],
        [[9 , 1 , 2, 1, 5, 10], [6, 1 , 7 , 9, 3, 0], [2, 1 , 4, 8 , 5, 7], [5, 7 , 0, 4, 3, 2], [2 , 7, 5, 8, 5, 6]],
        [[2 , 8 , 5, 9, 1, 5 ], [5, 10, 6 , 8, 0, 1], [0, 10, 0, 8 , 6, 7], [5, 1 , 4, 9, 3, 0], [1 , 1, 4, 3, 9, 4]]
      ].toTensor

      check: argmin(a, 0) == [
        [[0, 1, 1, 2, 1, 0], [1, 2, 3, 1, 3, 2], [3, 1, 3, 0, 1, 1], [2, 3, 2, 2, 2, 3], [3, 3, 0, 0, 1, 1]]
      ].toTensor

      check: argmin(a, 1) == [
        [[0, 1, 4, 2, 1, 3]], [[1, 0, 0, 1, 0, 2]], [[2, 0, 3, 0, 1, 1]], [[2, 3, 2, 4, 1, 3]]
      ].toTensor

      check: argmin(a, 2) == [
        [[0], [1], [3], [5], [3]], [[1], [3], [1], [2], [4]], [[1], [5], [1], [2], [0]], [[4], [4], [0], [5], [0]]
      ].toTensor

    block:
      let a = [
        [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]],
        [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]
      ].toTensor

      check: argmin(a, 0) == [
        [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
      ].toTensor

      check: argmin(a, 1) == [
        [[0, 0, 0, 0]], [[0, 0, 0, 0]]
      ].toTensor

      check: argmin(a, 2) == [
        [[0], [0], [0]], [[0], [0], [0]]
      ].toTensor

      test "cumsum":
        let a = [[0, 4, 7],
                 [1, 9, 5],
                 [3, 4, 1]].toTensor

        check: cumsum(a, 0) == [[0, 4,  7],
                                [1, 13, 12],
                                [4, 17, 13]].toTensor

        check: cumsum(a, 1) == [[0, 4, 11],
                                [1, 10, 15],
                                [3, 7, 8]].toTensor
      test "cumprod":
        let a = [[0, 4, 7],
                 [1, 9, 5],
                 [3, 4, 1]].toTensor

        check: cumprod(a, 0) == [[0, 4,   7],
                                 [0, 36,  35],
                                 [0, 144, 35]].toTensor

        check: cumprod(a, 1) == [[0, 0, 0],
                                 [1, 9, 45],
                                 [3, 12, 12]].toTensor

    test "diff":
      let a = arange(12).reshape([3, 4])
      block:
        # Test diffs along the default axis
        let expected_diff1_axis1 = ones[int](3, 3)
        let expected_diff2_axis1 = zeros[int](3, 2)
        check a.diff_discrete(0) == a
        check a.diff_discrete == expected_diff1_axis1
        check a.diff_discrete(2) == expected_diff2_axis1
      block:
        # Test diffs along a different axis
        let expected_diff1_axis0 = 4 * ones[int](2, 4)
        let expected_diff2_axis0 = zeros[int](1, 4)
        check a.diff_discrete(0, axis=0) == a
        check a.diff_discrete(axis=0) == expected_diff1_axis0
        check a.diff_discrete(2, axis=0) == expected_diff2_axis0
      block:
        # Test boolean diffs
        let b = [true, true, false, false, true].toTensor
        let expected_bool_diff1 = [false, true, false, true].toTensor
        let expected_bool_diff2 = [true, true, true].toTensor
        check b.diff_discrete(0) == b
        check b.diff_discrete() == expected_bool_diff1
        check b.diff_discrete(2) == expected_bool_diff2

  test "unwrap_period":
    block:
      # Single dimension unwrap_period
      let phase_deg = (linspace(0, 720, 9) mod 360).asType(int) -. 180
      let expected = linspace(0, 720, 9).asType(int) -. 180
      check unwrap_period(phase_deg, period=360) == expected

      # Check that unwrap_period also works with floats
      check unwrap_period(phase_deg.asType(float), period=360.0) == expected.asType(float)

      let phase_rad = (linspace(0.0, 4*PI, 9).floorMod(2*PI)) -. PI
      let expected_rad = linspace(0.0, 4*PI, 9) -. PI
      check unwrap_period(phase_rad, period=2*PI) == expected_rad
      check unwrap_period(phase_rad) == expected_rad
    block:
      # Multiple dimension unwrap_period through axis
      let a = arange(0, 60*5, 5).reshape([3,4,5])
      let expected_axis0 = [[[ 0,  5, 10, 15, 20],
                             [25, 30, 35, 40, 45],
                             [50, 55, 60, 65, 70],
                             [75, 80, 85, 90, 95]],
                            [[-2,  3,  8, 13, 18],
                             [23, 28, 33, 38, 43],
                             [48, 53, 58, 63, 68],
                             [73, 78, 83, 88, 93]],
                            [[-4,  1,  6, 11, 16],
                             [21, 26, 31, 36, 41],
                             [46, 51, 56, 61, 66],
                             [71, 76, 81, 86, 91]]].toTensor
      let expected_axis2 = [[[ 0, -1, -2, -3, -4],
                             [25, 24, 23, 22, 21],
                             [50, 49, 48, 47, 46],
                             [75, 74, 73, 72, 71]],
                            [[100,  99,  98,  97,  96],
                             [125, 124, 123, 122, 121],
                             [150, 149, 148, 147, 146],
                             [175, 174, 173, 172, 171]],
                            [[200, 199, 198, 197, 196],
                             [225, 224, 223, 222, 221],
                             [250, 249, 248, 247, 246],
                             [275, 274, 273, 272, 271]]].toTensor
      check unwrap_period(a, axis=0, period=6) == expected_axis0
      check unwrap_period(a, axis=2, period=6) == expected_axis2
      # When the axis is not specified, the innermost axis is used
      check unwrap_period(a, period=6) == expected_axis2
      # Check that unwrap_period also works with floats
      check unwrap_period(a.asType(float), period=6.0) == expected_axis2.asType(float)

  test "Nonzero":
    block:
      let a = [[3, 0, 0], [0, 4, 0], [5, 6, 0]].toTensor()
      let exp = [[0, 1, 2, 2], [0, 1, 0, 1]].toTensor
      check a.nonzero == exp

    block:
      let a = [[1, 2, 3], [4, 5, 6], [7, 8, 9]].toTensor
      let mask = a >. 3
      let exp = [[1, 1, 1, 2, 2, 2], [0, 1, 2, 0, 1, 2]].toTensor
      check nonzero(mask) == exp

main()
GC_fullCollect()
