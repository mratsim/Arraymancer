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
import unittest, math, sequtils
import complex except Complex64, Complex32

testSuite "Creating a new Tensor":
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

    let t4 = @[complex64(1.0,0.0),complex64(2.0,0.0),complex64(3.0,0.0)].toTensor()
    check: t4.shape == [3]
    check: t4.rank == 1

    let u = @[@[1.0, -1, 2],@[0.0, -1]]

    when compileOption("boundChecks") and not defined(openmp):
      expect(IndexDefect):
        discard u.toTensor()
    else:
      echo "Bound-checking is disabled or OpenMP is used. The incorrect seq shape test has been skipped."

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
    block:
      let t = zeros[Complex[float64]]([4,4,4])
      for v in t.items:
        check v == complex64(0.0,0.0)

  test "Ones":
    block:
      let t = ones[float]([4,4,4])
      for v in t.items:
        check v == 1.0f
    block:
      let t = ones[int]([4,4,4])
      for v in t.items:
        check v == 1
    block:
      let t = ones[Complex[float64]]([4,4,4])
      for v in t.items:
        check v == complex64(1.0, 0.0)

  test "Filled new tensor":
    block:
      let t = newTensorWith([4,4,4], 2.0f)
      for v in t.items:
        check v == 2.0f
    block:
      let t = newTensorWith([4,4,4], 2)
      for v in t.items:
        check v == 2
    block:
      let t = newTensorWith([4,4,4], complex64(2.0, 0.0))
      for v in t.items:
        check v == complex64(2.0, 0.0)

  test "arange - initialization from (start, stop, step)":
    # From Numpy docs
    block:
      let t = arange(3)
      check: t == [0,1,2].toTensor()
    block:
      let t = arange(3.0)
      check: t == [float64 0,1,2].toTensor()
    block:
      let t = arange(3,7)
      check: t == [3,4,5,6].toTensor()
    block:
      let t = arange(3,7,2)
      check: t == [3,5].toTensor()
    # From PyTorch docs
    block:
      let t = arange(1.0,2.5,0.5)
      check: t == [1.0,1.5,2.0].toTensor()

  test "linspace - initialization from (start, stop, num)":
    block:
      # with endpoint
      let exp = @[2.00, 2.25, 2.5, 2.75, 3.00].toTensor()
      let t = linspace(2.0, 3.0, num = 5)
      check t.size == 5
      for i in 0 ..< t.size:
        check round(t[i], places = 2) == round(exp[i], places = 2)
    block:
      let exp = @[2.00, 2.20, 2.40, 2.60, 2.80].toTensor()
      let t = linspace(2.0, 3.0, num = 5, endpoint = false)
      check t.size == 5
      for i in 0 ..< t.size:
        check round(t[i], places = 2) == round(exp[i], places = 2)
    block:
      # start larger than stop
      let exp = @[5.0, 4.0, 3.0, 2.0, 1.0, 0.0].toTensor()
      let t = linspace(5.0, 0.0, num = 6)
      check t.size == 6
      for i in 0 ..< t.size:
        check round(t[i], places = 2) == round(exp[i], places = 2)

  test "logspace - initialization from (start, stop, num)":
    block:
      # with endpoint
      let exp = @[1e0, 1e1, 1e2, 1e3, 1e4]
      let t = logspace(0, 4, num = 5)
      check t.size == 5
      for i in 0 ..< t.size:
        check round(t[i], places = 2) == round(exp[i], places = 2)
    block:
      let exp = @[1e0, 1e1, 1e2, 1e3, 1e4]
      let t = logspace(0, 5, num = 5, endpoint = false)
      check t.size == 5
      for i in 0 ..< t.size:
        check round(t[i], places = 2) == round(exp[i], places = 2)

  test "Random tensor":
    # Check that randomTensor doesn't silently convert float32 to float64
    let a = randomTensor([3, 4], 100'f32)

    check: a[0,0] is float32
  # TODO add tests for randomTensor

  test "Random sampled tensors":
    let source = [1, 3, 7, 15, 31]
    let t = randomTensor([10, 10], sample_source = source)

    for val in t:
      check(val in source)

  test "Random normal tensor":
    for i in 0..<4:
      let t = randomNormalTensor[float32](1000)
      check: abs(t.mean()) <= 2e-1
      check: abs(t.std() - 1.0) <= 2e-1
