# Copyright 2017-2020 Mamy André-Ratsimbazafy
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
import unittest

suite "Selectors":
  test "Index_select (Numpy take)":
    block: # Numpy
      let a = [4, 3, 5, 7, 6, 8].toTensor
      let indices = [0, 1, 4].toTensor

      check: a.index_select(axis = 0, indices) == [4, 3, 6].toTensor

    block: # PyTorch
      let x =  [[ 0.1427,  0.0231, -0.5414, -1.0009],
                [-0.4664,  0.2647, -0.1228, -1.1068],
                [-1.1734, -0.6571,  0.7230, -0.6004]].toTensor

      let indices = [0, 2].toTensor

      let ax0 =  [[ 0.1427,  0.0231, -0.5414, -1.0009],
                  [-1.1734, -0.6571,  0.7230, -0.6004]].toTensor
      let ax1 =  [[ 0.1427, -0.5414],
                  [-0.4664, -0.1228],
                  [-1.1734,  0.7230]].toTensor

      check:
        x.index_select(axis = 0, indices) == ax0
        x.index_select(axis = 1, indices) == ax1

    # ------------------------------------------------------
    # Selection with regular arrays/sequences

    block: # Numpy
      let a = [4, 3, 5, 7, 6, 8].toTensor
      let indices = [0, 1, 4]

      check: a.index_select(axis = 0, indices) == [4, 3, 6].toTensor

    block: # PyTorch
      let x =  [[ 0.1427,  0.0231, -0.5414, -1.0009],
                [-0.4664,  0.2647, -0.1228, -1.1068],
                [-1.1734, -0.6571,  0.7230, -0.6004]].toTensor

      let indices = [0, 2]

      let ax0 =  [[ 0.1427,  0.0231, -0.5414, -1.0009],
                  [-1.1734, -0.6571,  0.7230, -0.6004]].toTensor
      let ax1 =  [[ 0.1427, -0.5414],
                  [-0.4664, -0.1228],
                  [-1.1734,  0.7230]].toTensor

      check:
        x.index_select(axis = 0, indices) == ax0
        x.index_select(axis = 1, indices) == ax1

  test "Masked_select":
    block: # Numpy reference doc
           # https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html#boolean-array-indexing
           # select non NaN
      # x = np.array([[1., 2.], [np.nan, 3.], [np.nan, np.nan]])
      # x[~np.isnan(x)]
      # array([ 1.,  2.,  3.])
      let x = [[1.0, 2.0],
               [Nan, 3.0],
               [Nan, Nan]].toTensor

      let r = x.masked_select(x.isNotNan)

      let expected = [1.0, 2.0, 3.0].toTensor()
      check: r == expected

    block: # with regular arrays/sequences
      let x = [[1.0, 2.0],
               [Nan, 3.0],
               [Nan, Nan]].toTensor

      let r = x.masked_select(
          [[true,  true],
           [false, true],
           [false, false]]
        )

      let expected = [1.0, 2.0, 3.0].toTensor()
      check: r == expected

  test "Masked_fill":
    block: # Numpy reference doc
           # https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html#boolean-array-indexing
           # select non NaN
      # x = np.array([[1., 2.], [np.nan, 3.], [np.nan, np.nan]])
      # x[np.isnan(x)] = -1
      # x
      # np.array([[1., 2.], [-1, 3.], [-1, -1]])
      var x = [[1.0, 2.0],
               [Nan, 3.0],
               [Nan, Nan]].toTensor

      x.masked_fill(x.isNan, -1.0)

      let expected = [[1.0, 2.0], [-1.0, 3.0], [-1.0, -1.0]].toTensor()
      check: x == expected

    block: # with regular arrays/sequences
      var x = [[1.0, 2.0],
               [Nan, 3.0],
               [Nan, Nan]].toTensor

      x.masked_fill(
        [[false,  false],
         [true, false],
         [true, true]],
        -1.0
      )

      let expected = [[1.0, 2.0], [-1.0, 3.0], [-1.0, -1.0]].toTensor()
      check: x == expected

  test "Masked_axis_select":
    block: # Numpy reference doc
           # https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html#boolean-array-indexing
           # select all rows which sum up to less or equal two
      # x = np.array([[0, 1], [1, 1], [2, 2]])
      # rowsum = x.sum(-1)
      # print(rowsum)
      # print(x[rowsum <= 2, :])
      # array([[0, 1],
      #        [1, 1]]
      let x = [[0, 1],
               [1, 1],
               [2, 2]].toTensor

      let rowsum = x.sum(axis = 1)
      let cond = rowsum <=. 2
      let r = x.masked_axis_select(cond.squeeze(), axis = 0)

      let expected = [[0, 1],
                      [1, 1]].toTensor

      check: r == expected

    block: # With regular arrays/sequences

      let x = [[0, 1],
               [1, 1],
               [2, 2]].toTensor

      let rowsum = x.sum(axis = 1)
      let cond = [true,
                  true,
                  false]
      let r = x.masked_axis_select(cond, axis = 0)

      let expected = [[0, 1],
                      [1, 1]].toTensor

  test "Masked_axis_fill":
    block: # Numpy
           # Fill all columns which sum up to greater than 1
           # with -10
      # import numpy as np
      # a = np.array([[-1, -2, 1], [1, 2, 0], [1, -1, 1]])
      # print(a.sum(axis=0) > 1)
      # a[:, a.sum(axis=0) > 1] = -10
      # print(a)
      var a = [[-1, -2, 1],
               [ 1,  2, 0],
               [ 1, -1, 1]].toTensor

      let cond = squeeze(a.sum(axis = 0) >. 1)
      a.masked_axis_fill(cond, axis = 1, -10)

      let expected = [[-1, -2, -10],
                      [ 1,  2, -10],
                      [ 1, -1, -10]].toTensor

      check: a == expected

    block: # With regular arrays/sequences
      var a = [[-1, -2, 1],
               [ 1,  2, 0],
               [ 1, -1, 1]].toTensor

      let cond = [false, false, true]
      a.masked_axis_fill(cond, axis = 1, -10)

      let expected = [[-1, -2, -10],
                      [ 1,  2, -10],
                      [ 1, -1, -10]].toTensor

      check: a == expected

    block: # Fill with tensor
      # import numpy as np
      # a = np.array([[-1, -2, 1], [1, 2, 0], [1, -1, 1]])
      # print(a.sum(axis=0) > 1)
      # a[:, a.sum(axis=0) > 1] = np.array([-10, -20, -30])[:, np.newaxis]
      # print(a)

      var a = [[-1, -2, 1],
               [ 1,  2, 0],
               [ 1, -1, 1]].toTensor

      let b = [-10, -20, -30].toTensor.unsqueeze(1)

      let cond = squeeze(a.sum(axis = 0) >. 1)
      a.masked_axis_fill(cond, axis = 1, b)

      let expected = [[-1, -2, -10],
                      [ 1,  2, -20],
                      [ 1, -1, -30]].toTensor

      check: a == expected

    block: # With regular arrays/sequences
      var a = [[-1, -2, 1],
               [ 1,  2, 0],
               [ 1, -1, 1]].toTensor

      let b = [-10, -20, -30].toTensor.unsqueeze(1)

      let cond = [false, false, true]
      a.masked_axis_fill(cond, axis = 1, b)

      let expected = [[-1, -2, -10],
                      [ 1,  2, -20],
                      [ 1, -1, -30]].toTensor

      check: a == expected

  test "Masked_fill_along_axis":
    block:
      var a = [[-1, -2, 1],
               [ 1,  2, 0],
               [ 1, -1, 1]].toTensor

      a.masked_fill_along_axis(a.sum(axis = 0) >. 1, axis = 0, -10)

      let expected = [[-1, -2, -10],
                      [ 1,  2, -10],
                      [ 1, -1, -10]].toTensor

      check: a == expected

    block: # Apply a 2D mask on a RGB image
      var checkered = [
        [ # Red
          [uint8 255, 255, 255, 255],
          [uint8 255, 255, 255, 255],
          [uint8 255, 255, 255, 255],
          [uint8 255, 255, 255, 255],
        ],
        [ # Green
          [uint8 255, 255, 255, 255],
          [uint8 255, 255, 255, 255],
          [uint8 255, 255, 255, 255],
          [uint8 255, 255, 255, 255],
        ],
        [ # Blue
          [uint8 255, 255, 255, 255],
          [uint8 255, 255, 255, 255],
          [uint8 255, 255, 255, 255],
          [uint8 255, 255, 255, 255],
        ]
      ].toTensor()

      var mask = [
        [false, true, false, true],
        [true, false, true, false],
        [false, true, false, true],
        [true, false, true, false]
      ].toTensor().unsqueeze(0)

      var expected = [
        [ # Red
          [uint8 255, 0, 255, 0],
          [uint8 0, 255, 0, 255],
          [uint8 255, 0, 255, 0],
          [uint8 0, 255, 0, 255],
        ],
        [ # Green
          [uint8 255, 0, 255, 0],
          [uint8 0, 255, 0, 255],
          [uint8 255, 0, 255, 0],
          [uint8 0, 255, 0, 255],
        ],
        [ # Blue
          [uint8 255, 0, 255, 0],
          [uint8 0, 255, 0, 255],
          [uint8 255, 0, 255, 0],
          [uint8 0, 255, 0, 255],
        ]
      ].toTensor()

      checkered.masked_fill_along_axis(mask, axis = 0, 0'u8)

      check: checkered == expected
