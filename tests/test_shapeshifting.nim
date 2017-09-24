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

suite "Shapeshifting":
  test "Contiguous conversion":
    let a = [7, 4, 3, 1, 8, 6, 8, 1, 6, 2, 6, 6, 2, 0, 4, 3, 2, 0].toTensor.reshape([3,6])

    # Tensor of shape 3x6 of type "int" on backend "Cpu"
    # |7      4       3       1       8       6|
    # |8      1       6       2       6       6|
    # |2      0       4       3       2       0|

    let b = a.asContiguous()
    check: b.toRawSeq == @[7, 4, 3, 1, 8, 6, 8, 1, 6, 2, 6, 6, 2, 0, 4, 3, 2, 0]

    # a is already contiguous, even if wrong layout.
    # Nothing should be done
    let c = a.asContiguous(colMajor)
    check: c.toRawSeq == @[7, 4, 3, 1, 8, 6, 8, 1, 6, 2, 6, 6, 2, 0, 4, 3, 2, 0]

    # force parameter has been used.
    # Layout will change even if a was contiguous
    let d = a.asContiguous(colMajor, force = true)
    check: d.toRawSeq == @[7, 8, 2, 4, 1, 0, 3, 6, 4, 1, 2, 3, 8, 6, 2, 6, 6, 0]


    # Now test with a non contiguous tensor
    let u = a[_,0..1]
    check: u.toRawSeq == @[7, 4, 3, 1, 8, 6, 8, 1, 6, 2, 6, 6, 2, 0, 4, 3, 2, 0]
    check: u == [7,4,8,1,2,0].toTensor.reshape([3,2])

    check: u.asContiguous.toRawSeq == @[7,4,8,1,2,0]
    check: u.asContiguous(colMajor).toRawSeq == @[7,8,2,4,1,0]

  test "Reshape":
    let a = toSeq(1..4).toTensor().reshape(2,2)
    check: a == [[1,2],
                 [3,4]].toTensor()

  test "Unsafe reshape":
    block:
      let a = toSeq(1..4).toTensor()
      var a_view = a.unsafeReshape(2,2)
      check: a_view == [[1,2],[3,4]].toTensor()
      a_view[_, _] = 0
      check: a == [0,0,0,0].toTensor()

    # on slices
    block:
      # not that 'a' here a let variable, however
      # unsafeView and unsafeReshape allow us to
      # modify its elements value
      let a = toSeq(1..4).toTensor()
      var a_view = a.unsafeSlice(1..2).unsafeReshape(1,2)
      check: a_view == [[2,3]].toTensor()
      a_view[_, _] = 0
      check: a == [1,0,0,4].toTensor()

  test "Concatenation":
    let a = toSeq(1..4).toTensor().reshape(2,2)

    let b = toSeq(5..8).toTensor().reshape(2,2)

    check: concat(a,b, axis = 0) == [[1,2],
                                     [3,4],
                                     [5,6],
                                     [7,8]].toTensor()
    check: concat(a,b, axis = 1) == [[1,2,5,6],
                                     [3,4,7,8]].toTensor()

  test "Squeeze":
    block:
      let a = toSeq(1..12).toTensor().reshape(1,3,1,2,1,1,2)
      let b = a.squeeze

      check b == toSeq(1..12).toTensor().reshape(3,2,2)

    block: # With slices
      let a = toSeq(1..12).toTensor().reshape(1,3,1,4)
      let b = a[0, 0, 0, 3..0|-1].squeeze

      check b == [4,3,2,1].toTensor

    block: # Single axis
      let a = toSeq(1..12).toTensor().reshape(1,3,1,4)
      let b = a.squeeze(2)

      check b == toSeq(1..12).toTensor().reshape(1,3,4)

    block: # Single axis with slices
      let a = toSeq(1..12).toTensor().reshape(1,3,1,4)
      let b = a[0, 0..1, 0, 3..0|-1].squeeze(0)

      check b == [4, 3, 2, 1, 8, 7, 6, 5].toTensor.reshape(2,1,4)

  test "To tensor reshape":
    block:
      var s = @[1,2,3,4]
      var a = s.unsafeToTensorReshape([2,2])
      check a == [[1,2],[3,4]].toTensor()
      s[0] = 0
      check a == [[0,2],[3,4]].toTensor()

  test "Unsqueeze":
    block:
      let a = toSeq(1..12).toTensor().reshape(3,4)
      let b = a.unsqueeze(0)
      let c = a.unsqueeze(1)
      let d = a.unsqueeze(2)

      check a.reshape(1,3,4).strides == b.strides
      check a.reshape(3,1,4).strides == c.strides
      check a.reshape(3,4,1).strides == d.strides
      check b == toSeq(1..12).toTensor().reshape(1,3,4)
      check c == toSeq(1..12).toTensor().reshape(3,1,4)
      check d == toSeq(1..12).toTensor().reshape(3,4,1)

    block: # With slices
      let a = toSeq(1..12).toTensor().reshape(3,4)
      let b = a[0..1, ^2..0|-1].unsqueeze(0)
      let c = a[0..1, ^2..0|-1].unsqueeze(1)
      let d = a[0..1, ^2..0|-1].unsqueeze(2)

      check b == [[[3,2,1],[7,6,5]]].toTensor
      check c == [[[3,2,1]],[[7,6,5]]].toTensor
      check d == [[[3],[2],[1]],[[7],[6],[5]]].toTensor

  test "Stack tensors":
    let a = [[1,2,3].toTensor(),[4,5,6].toTensor()]
    check a.stack() == [[1,2,3],[4,5,6]].toTensor()
    check a.stack(1) == [[1,4],[2,5],[3,6]].toTensor()

    let b = [[[1,2],[3,4]].toTensor(),[[4,5],[6,7]].toTensor()]
    check b.stack()  == [[[1,2],[3,4]],[[4,5],[6,7]]].toTensor()
    check b.stack(1) == [[[1,2],[4,5]],[[3,4],[6,7]]].toTensor()
    check b.stack(2) == [[[1,4],[2,5]],[[3,6],[4,7]]].toTensor()

