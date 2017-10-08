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
import unittest, future, sequtils

suite "Shapeshifting":
  test "Reshape":
    let a = toSeq(1..4).toTensor(Cpu).reshape(2,2)
    check: a == [[1,2],
                 [3,4]].toTensor(Cpu)

  test "Concatenation":
    let a = toSeq(1..4).toTensor(Cpu).reshape(2,2)

    let b = toSeq(5..8).toTensor(Cpu).reshape(2,2)

    check: concat(a,b, axis = 0) == [[1,2],
                                     [3,4],
                                     [5,6],
                                     [7,8]].toTensor(Cpu)
    check: concat(a,b, axis = 1) == [[1,2,5,6],
                                     [3,4,7,8]].toTensor(Cpu)