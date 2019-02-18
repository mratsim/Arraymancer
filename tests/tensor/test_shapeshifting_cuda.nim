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

import ../../src/arraymancer
import unittest, sugar, sequtils

suite "CUDA: Shapeshifting":
  ## Note: by default (momentarily), CudaTensors are column-major
  test "Contiguous conversion":
    let a = [7, 4, 3, 1, 8, 6,
             8, 1, 6, 2, 6, 6,
             2, 0, 4, 3, 2, 0].toTensor.reshape([3,6]).astype(float).cuda

    # Tensor of shape 3x6 of type "int" on backend "Cpu"
    # |7      4       3       1       8       6|
    # |8      1       6       2       6       6|
    # |2      0       4       3       2       0|

    let b = a.asContiguous()
    check: b.cpu.toRawSeq == @[7.0, 8, 2, 4, 1, 0, 3, 6, 4, 1, 2, 3, 8, 6, 2, 6, 6, 0]

    # a is already contiguous, even if wrong layout.
    # Nothing should be done
    let c = a.asContiguous(colMajor)
    check: c.cpu.toRawSeq == @[7.0, 8, 2, 4, 1, 0, 3, 6, 4, 1, 2, 3, 8, 6, 2, 6, 6, 0]

    # force parameter has been used.
    # Layout will change even if a was contiguous
    let d = a.asContiguous(colMajor, force = true)
    check: d.cpu.toRawSeq == @[7.0, 8, 2, 4, 1, 0, 3, 6, 4, 1, 2, 3, 8, 6, 2, 6, 6, 0]


    # Now test with a non-contiguous tensor
    let u = a[_,0..1]
    check: u.cpu.toRawSeq == @[7.0, 8, 2, 4, 1, 0, 3, 6, 4, 1, 2, 3, 8, 6, 2, 6, 6, 0]
    check: u.cpu == [7.0,4,8,1,2,0].toTensor.reshape([3,2])

    check: u.asContiguous(rowMajor, force=true).cpu.toRawSeq == @[7.0,4,8,1,2,0]

  test "Unsafe reshape":
    block:
      let a = toSeq(1..4).toTensor().astype(float).cuda
      var a_view = a.reshape(2,2)
      check: a_view.cpu == [[1.0,2],[3.0,4]].toTensor()

      # TODO
      # a_view[_, _] = 0.0
      # check: a.cpu == [0.0,0,0,0].toTensor()

    # on slices
    block:
      # not that 'a' here a let variable, however
      # unsafeView and reshape allow us to
      # modify its elements value
      let a = toSeq(1..4).toTensor().astype(float).cuda
      var a_view = a[1..2].reshape(1,2)
      check: a_view.cpu == [[2.0,3]].toTensor()

      # TODO: pending slice assignation
      # a_view[_, _].cpu = 0
      # check: a.cpu == [1.0,0,0,4].toTensor()
