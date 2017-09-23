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

import ../src/arraymancer
import unittest, future, sequtils

suite "Shapeshifting - broadcasting":
  test "Explicit broadcasting":
    block:
      let a = 2.bc([3,2])
      check a == [[2,2],
                  [2,2],
                  [2,2]].toTensor()

    block:
      let a = toSeq(1..2).toTensor().reshape(1,2)
      let b = a.bc([2,2])
      check b == [[1,2],
                  [1,2]].toTensor()

    block:
      let a = toSeq(1..2).toTensor().reshape(2,1)
      let b = a.bc([2,2])
      check b == [[1,1],
                  [2,2]].toTensor()

  test "Implicit broadcasting - addition":
    block:
      let a = [0, 10, 20, 30].toTensor().reshape(4,1)
      let b = [0, 1, 2].toTensor().reshape(1,3)

      check: a .+ b == [[0, 1, 2],
                        [10, 11, 12],
                        [20, 21, 22],
                        [30, 31, 32]].toTensor