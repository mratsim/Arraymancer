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

