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
      let cond = rowsum .<= 2
      let r = x.masked_axis_select(cond.squeeze(), axis = 0)

      let expected = [[0, 1],
                      [1, 1]].toTensor

      check: r == expected

  test "masked_axis_fill":
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

      let cond = squeeze(a.sum(axis = 0) .> 1)
      a.masked_axis_fill(cond, axis = 1, -10)

      let expected = [[-1, -2, -10],
                      [ 1,  2, -10],
                      [ 1, -1, -10]].toTensor

      check: a == expected

  test "Masked_fill_along_axis":
    block:
      var a = [[-1, -2, 1],
               [ 1,  2, 0],
               [ 1, -1, 1]].toTensor

      a.masked_fill_along_axis(a.sum(axis = 0) .> 1, axis = 0, -10)

      let expected = [[-1, -2, -10],
                      [ 1,  2, -10],
                      [ 1, -1, -10]].toTensor

      check: a == expected
