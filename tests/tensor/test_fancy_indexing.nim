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

suite "Fancy indexing":
  # x = np.array([[ 4, 99,  2],
  #               [ 3,  4, 99],
  #               [ 1,  8,  7],
  #               [ 8,  6,  8]])

  let x = [[ 4, 99,  2],
            [ 3,  4, 99],
            [ 1,  8,  7],
            [ 8,  6,  8]].toTensor()

  test "Index selection via fancy indexing":
    block: # print(x[:, [0, 2]])
      let r = x[_, [0, 2]]

      let exp = [[ 4,  2],
                 [ 3, 99],
                 [ 1,  7],
                 [ 8,  8]].toTensor()

      check: r == exp

    block: # print(x[[1, 3], :])
      let r = x[[1, 3], _]

      let exp = [[3, 4, 99],
                 [8, 6,  8]].toTensor()

      check: r == exp
