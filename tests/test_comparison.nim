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

import ../arraymancer
import unittest, math


suite "Testing tensor comparison":
    test "Testing for [1..^2, 1..3] slicing":
        const
            a = @[1, 2, 3, 4, 5]
            b = @[1, 2, 3, 4, 5]

        var
            vandermonde: seq[seq[int]]
            row: seq[int]

        vandermonde = newSeq[seq[int]]()

        for i, aa in a:
            row = newSeq[int]()
            vandermonde.add(row)
            for j, bb in b:
                vandermonde[i].add(aa^bb)

        let t_van = vandermonde.toTensor(Cpu)

        # Tensor of shape 5x5 of type "int" on backend "Cpu"
        # |1      1       1       1       1|
        # |2      4       8       16      32|
        # |3      9       27      81      243|
        # |4      16      64      256     1024|
        # |5      25      125     625     3125|

        let test = @[@[4, 8, 16], @[9, 27, 81], @[16, 64, 256]]
        let t_test = test.toTensor(Cpu)
        
        check: t_van[1..^2,1..3] == t_test
        check: t_van[1..3,1..3] == t_test