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


testSuite "CUDA: Testing indexing and slice syntax":
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

  let t_van = vandermonde.toTensor().astype(float32).cuda()

  # TODO: Indexing (not slicing) is not yet supported on CUDA
  #  # Tensor of shape 5x5 of type "int" on backend "Cpu"
  #  # |1      1       1       1       1|
  #  # |2      4       8       16      32|
  #  # |3      9       27      81      243|
  #  # |4      16      64      256     1024|
  #  # |5      25      125     625     3125|
  #  test "Basic indexing - foo[2, 3]":
  #    check: t_van[2, 3] == 81
  #
  #  test "Basic indexing - foo[1+1, 2*2*1]":
  #    check: t_van[1+1, 2*2*1] == 243
  #
  #  ## TODO: The following checks fails because return value
  #  ## is a Tensor not a scalar
  #  # test "Indexing from end - foo[^1, 3]":
  #  #     check: t_van[^1, 3] == 625
  #  # test "Indexing from end - foo[^(1*2), 3]":
  #  #     check: t_van[^1, 3] == 256
  test "Basic slicing - foo[1..2, 3]":
    let test = @[@[16],@[81]]
    check: t_van[1..2, 3].cpu == test.toTensor().astype(float32)

  test "Basic slicing - foo[1+1..4, 3-2..2]":
    let test = @[@[9,27],@[16, 64],@[25, 125]]
    check: t_van[1+1..4, 3-2..2].cpu == test.toTensor().astype(float32)

  test "Span slices - foo[_, 3]":
    let test = @[@[1],@[16],@[81],@[256],@[625]]
    check: t_van[_, 3].cpu == test.toTensor().astype(float32)

  test "Span slices - foo[1.._, 3]":
    let test = @[@[16],@[81],@[256],@[625]]
    check: t_van[1.._, 3].cpu == test.toTensor().astype(float32)

    ## Check with extra operators
    check: t_van[0+1.._, 3].cpu == test.toTensor().astype(float32)

  test "Span slices - foo[_..3, 3]":
    let test = @[@[1],@[16],@[81],@[256]]
    check: t_van[_..3, 3].cpu == test.toTensor().astype(float32)

    ## Check with extra operators
    check: t_van[_..5-2, 3].cpu == test.toTensor().astype(float32)

  test "Span slices - foo[_.._, 3]":
    let test = @[@[1],@[16],@[81],@[256],@[625]]
    check: t_van[_.._, 3].cpu == test.toTensor().astype(float32)

  test "Stepping - foo[1..3|2, 3]":
    let test = @[@[16],@[256]]
    check: t_van[1..3|2, 3].cpu == test.toTensor().astype(float32)
    check: t_van[1..3|+2, 3].cpu == test.toTensor().astype(float32)
    check: t_van[1*(0+1)..2+1|(5-3), 3].cpu == test.toTensor().astype(float32)

  test "Span stepping - foo[_.._|2, 3]":
    let test = @[@[1],@[81],@[625]]
    check: t_van[_.._|2, 3].cpu == test.toTensor().astype(float32)

  test "Span stepping - foo[_.._|+2, 3]":
    let test = @[@[1],@[81],@[625]]
    check: t_van[_.._|+2, 3].cpu == test.toTensor().astype(float32)

  test "Span stepping - foo[1.._|1, 2..3]":
    let test = @[@[8, 16],@[27, 81],@[64, 256], @[125, 625]]
    check: t_van[1.._|1, 2..3].cpu == test.toTensor().astype(float32)

  test "Span stepping - foo[_..<4|2, 3]":
    let test = @[@[1],@[81]]
    check: t_van[_..<4|2, 3].cpu == test.toTensor().astype(float32)

  test "Slicing until at n from the end - foo[0..^4, 3]":
    let test = @[@[1],@[16]]
    check: t_van[0..^4, 3].cpu == test.toTensor().astype(float32)
    ## Check with extra operators
    check: t_van[0..^2+2, 3].cpu == test.toTensor().astype(float32)

  test "Span Slicing until at n from the end - foo[_..^2, 3]":
    let test = @[@[1],@[16],@[81],@[256]]
    check: t_van[_..^2, 3].cpu == test.toTensor().astype(float32)
    ## Check with extra operators
    check: t_van[_..^1+1, 3].cpu == test.toTensor().astype(float32)

  test "Stepped Slicing until at n from the end - foo[1..^1|2, 3]":
    let test = @[@[16],@[256]]
    check: t_van[1..^1|2, 3].cpu == test.toTensor().astype(float32)
    ## Check with extra operators
    check: t_van[1..^1|(1+1), 3].cpu == test.toTensor().astype(float32)

  test "Slice from the end - foo[^1..0|-1, 3]":
    let test = @[@[625],@[256],@[81],@[16],@[1]]
    check: t_van[^1..0|-1, 3].cpu == test.toTensor().astype(float32)
    ## Check with extra operators
    let test2 = @[@[256],@[81],@[16],@[1]]
    check: t_van[^(4-2)..0|-1, 3].cpu == test2.toTensor().astype(float32)

  when compileOption("boundChecks") and not defined(openmp):
    test "Slice from the end - expect non-negative step error - foo[^1..0, 3]":
      expect(IndexDefect):
        discard t_van[^1..0, 3]
  else:
    echo "Bound-checking is disabled or OpenMP is used. The non-negative step test has been skipped."

  test "Slice from the end - foo[^(2*2)..2*2, 3]":
    let test = @[@[16],@[81],@[256],@[625]]
    check: t_van[^(2*2)..2*2, 3].cpu == test.toTensor().astype(float32)

  test "Slice from the end - foo[^3..^2, 3]":
    let test = @[@[81],@[256]]
    check: t_van[^3..^2, 3].cpu == test.toTensor().astype(float32)
