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
import unittest, math, sequtils

suite "Creating a new Tensor":
  test "Creating from sequence":
    let t1 = @[1,2,3].toTensor()
    check: t1.shape == [3]
    check: t1.rank == 1

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

    let t2 = vandermonde.toTensor()
    check: t2.rank == 2
    check: t2.shape == [5, 5]

    let nest3 = @[
            @[
              @[1,2,3],
              @[1,2,3]
            ],
            @[
              @[3,2,1],
              @[3,2,1]
            ],
            @[
              @[4,4,5],
              @[4,4,4]
            ],
            @[
              @[6,6,6],
              @[6,6,6]
            ]
          ]

    let t3 = nest3.toTensor()
    check: t3.rank == 3
    check: t3.shape == [4, 2, 3] # 4 rows, 2 cols, 3 depth. depth indices moves the fastest. Same scheme as Numpy.

    let u = @[@[1.0, -1, 2],@[0.0, -1]]

    when compileOption("boundChecks"):
      expect(IndexError):
        discard u.toTensor()
    else:
      echo "Bound-checking is disabled. The incorrect seq shape test has been skipped."

  test "Check that Tensor shape is in row-by-column order":
    let s = @[@[1,2,3],@[3,2,1]]
    let t = s.toTensor()

    check: t.shape == [2,3]

    let u = newTensor[int](@[2,3])
    check: u.shape == [2,3]

    check: u.shape == t.shape

  test "Zeros":
    block:
      let t = zeros[float]([4,4,4])
      for v in t.items:
        check v == 0.0f
    block:
      let t = zeros[int]([4,4,4])
      for v in t.items:
        check v == 0

  test "Ones":
    block:
      let t = ones[float]([4,4,4])
      for v in t.items:
        check v == 1.0f
    block:
      let t = ones[int]([4,4,4])
      for v in t.items:
        check v == 1

  test "Filled new tensor":
    block:
      let t = newTensorWith([4,4,4], 2.0f)
      for v in t.items:
        check v == 2.0f
    block:
      let t = newTensorWith([4,4,4], 2)
      for v in t.items:
        check v == 2

  test "Random tensor":
    block:
      # Check that randomTensor doesn't silently convert float32 to float64
      let a = randomTensor([3, 4], 100'f32)

      check: a[0,0] is float32
  # TODO add tests for randomTensor

