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

import ../../src/arraymancer, ../testutils
import unittest, math
import complex except Complex64, Complex32

proc main() =
  suite "[Core] Testing aggregation functions":
    let t = [[0, 1, 2],
            [3, 4, 5],
            [6, 7, 8],
            [9, 10, 11]].toTensor()
    let t_c = t.astype(Complex[float64])

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
      check: t_c.sum(axis=0) == row_sum.astype(Complex[float64])
      check: t_c.sum(axis=1) == col_sum.astype(Complex[float64])

    test "Mean":
      check: t.astype(float).mean == 5.5 # Note: may fail due to float rounding
      check: t_c.mean == complex(5.5) # Note: may fail due to float rounding

      let row_mean = [[4.5, 5.5, 6.5]].toTensor()
      let col_mean = [[1.0],
                      [4.0],
                      [7.0],
                      [10.0]].toTensor()
      check: t.astype(float).mean(axis=0) == row_mean
      check: t.astype(float).mean(axis=1) == col_mean
      check: t_c.mean(axis=0) == row_mean.astype(Complex[float64])
      check: t_c.mean(axis=1) == col_mean.astype(Complex[float64])

    test "Product":
      let a = [[1,2,4],[8,16,32]].toTensor()
      let a_c = a.astype(Complex[float64])
      check: t.product() == 0
      check: a.product() == 32768
      check: a.astype(float).product() == 32768.0
      check: a.product(0) == [[8,32,128]].toTensor()
      check: a.product(1) == [[8],[4096]].toTensor()
      check: t_c.product() == complex(0.0)
      check: a_c.product() == complex(32768.0)
      check: a_c.product(0) == [[8,32,128]].toTensor().astype(Complex[float64])
      check: a_c.product(1) == [[8],[4096]].toTensor().astype(Complex[float64])

    test "Min":
      let a = [2,-1,3,-3,5,0].toTensor()
      check: a.min() == -3
      check: a.astype(float32).min() == -3.0f

      let b = [[1,2,3,-4],[0,4,-2,5]].toTensor()
      check: b.min(0) == [[0,2,-2,-4]].toTensor()
      check: b.min(1) == [[-4],[-2]].toTensor()
      check: b.astype(float32).min(0) == [[0.0f,2,-2,-4]].toTensor()
      check: b.astype(float32).min(1) == [[-4.0f],[-2.0f]].toTensor()

    test "Max":
      let a = [2,-1,3,-3,5,0].toTensor()
      check: a.max() == 5
      check: a.astype(float32).max() == 5.0f

      let b = [[1,2,3,-4],[0,4,-2,5]].toTensor()
      check: b.max(0) == [[1,4,3,5]].toTensor()
      check: b.max(1) == [[3],[5]].toTensor()
      check: b.astype(float32).max(0) == [[1.0f,4,3,5]].toTensor()
      check: b.astype(float32).max(1) == [[3.0f],[5.0f]].toTensor()

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

        check: cumprod(a, 0) == [[0, 4,   7],
                                 [0, 36,  35],
                                 [0, 144, 35]].toTensor
        
        check: cumprod(a, 1) == [[0, 0, 0],
                                 [1, 9, 45],
                                 [3, 12, 12]].toTensor


main()
GC_fullCollect()
