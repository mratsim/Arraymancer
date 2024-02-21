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
import unittest, sequtils
import complex except Complex64, Complex32

proc main() =
  suite "Shapeshifting":
    test "Contiguous conversion":
      block:
        let a = [7, 4, 3, 1, 8, 6, 8, 1, 6, 2, 6, 6, 2, 0, 4, 3, 2, 0].toTensor.reshape([3,6])
        let a_c = [7, 4, 3, 1, 8, 6, 8, 1, 6, 2, 6, 6, 2, 0, 4, 3, 2, 0].toTensor.reshape([3,6]).asType(Complex[float64])

        # Tensor of shape 3x6 of type "int" on backend "Cpu"
        # |7      4       3       1       8       6|
        # |8      1       6       2       6       6|
        # |2      0       4       3       2       0|

        let b = a.asContiguous()
        let b_c = a_c.asContiguous()
        check: b.toFlatSeq == @[7, 4, 3, 1, 8, 6, 8, 1, 6, 2, 6, 6, 2, 0, 4, 3, 2, 0]
        check: b_c.toFlatSeq == @[complex64(7'f64,0.0), complex64(4'f64,0.0), complex64(3'f64,0.0), complex64(1'f64,0.0), complex64(8'f64,0.0), complex64(6'f64,0.0), complex64(8'f64,0.0), complex64(1'f64,0.0), complex64(6'f64,0.0), complex64(2'f64,0.0), complex64(6'f64,0.0), complex64(6'f64,0.0), complex64(2'f64,0.0), complex64(0'f64,0.0), complex64(4'f64,0.0), complex64(3'f64,0.0), complex64(2'f64,0.0), complex64(0'f64,0.0)]

        # a is already contiguous, even if wrong layout.
        # Nothing should be done
        let c = a.asContiguous(colMajor)
        check: c.toFlatSeq == @[7, 4, 3, 1, 8, 6, 8, 1, 6, 2, 6, 6, 2, 0, 4, 3, 2, 0]

        # force parameter has been used.
        # Layout will change even if a was contiguous
        let d = a.asContiguous(colMajor, force = true)
        # this test needs `toRawSeq` due to the changed layout. `toFlatSeq` provides the
        # same as for `c` above!
        check: d.toRawSeq == @[7, 8, 2, 4, 1, 0, 3, 6, 4, 1, 2, 3, 8, 6, 2, 6, 6, 0]


        # # Now test with a non contiguous tensor
        #let u = a[_,0..1]
        #check: u.toFlatSeq == @[7, 4, 3, 1, 8, 6, 8, 1, 6, 2, 6, 6, 2, 0, 4, 3, 2, 0]
        #check: u == [7,4,8,1,2,0].toTensor.reshape([3,2])
        #
        #check: u.asContiguous.toFlatSeq == @[7,4,8,1,2,0]
        #check: u.asContiguous(colMajor).toFlatSeq == @[7,8,2,4,1,0]

      block: # Check Fortran order on 3d+ tensors
        let a = randomTensor([3, 1, 4, 4], 100) # [batch_size, color channel, height, width]
        check: a.strides == [16, 16, 4, 1]
        let b = a.asContiguous(colMajor, force = true)
        check: b.strides == [1, 3, 3, 12]

        check: a == b

    test "Reshape":
      let a = toSeq(1..4).toTensor().reshape(2,2)
      check: a == [[1,2],
                  [3,4]].toTensor()

    test "Unsafe reshape":
      block:
        let a = toSeq(1..4).toTensor()
        var a_view = a.reshape(2,2)
        check: a_view == [[1,2],[3,4]].toTensor()
        a_view[_, _] = 0
        check: a == [0,0,0,0].toTensor()

      # on slices
      block:
        # not that 'a' here a let variable, however
        # unsafeView and reshape allow us to
        # modify its elements value
        let a = toSeq(1..4).toTensor()
        var a_view = a[1..2].reshape(1,2)
        check: a_view == [[2,3]].toTensor()
        a_view[_, _] = 0
        check: a == [1,0,0,4].toTensor()

    test "Flatten":
      let a = [[1, 2], [3, 4]].toTensor()
      let b = a.flatten()
      check: b == [1,2,3,4].toTensor()

    test "Concatenation":
      let a = toSeq(1..4).toTensor().reshape(2,2)

      let b = toSeq(5..8).toTensor().reshape(2,2)

      check: concat(a,b, axis = 0) == [[1,2],
                                      [3,4],
                                      [5,6],
                                      [7,8]].toTensor()
      check: concat(a,b, axis = 1) == [[1,2,5,6],
                                      [3,4,7,8]].toTensor()

    test "Append":
      let a = toSeq(1..4).toTensor()
      let b = toSeq(5..8)
      let expected = [1,2,3,4,5,6,7,8].toTensor()

      check: a.append(5) == [1,2,3,4,5].toTensor()
      check: a.append(5, 6, 7, 8) == expected
      check: a.append(b) == expected
      check: a.append(b.toTensor()) == expected

    test "Squeeze":
      block:
        let a = toSeq(1..12).toTensor().reshape(3,1,2,1,1,2)
        let b = a.squeeze

        check: b == toSeq(1..12).toTensor().reshape(3,2,2)

      block: # With slices
        let a = toSeq(1..12).toTensor().reshape(1,3,1,4)
        let b = a[0, 0, 0, 3..0|-1].squeeze

        check: b == [4,3,2,1].toTensor

      block: # Single axis
        let a = toSeq(1..12).toTensor().reshape(1,3,1,4)
        let b = a.squeeze(2)

        check: b == toSeq(1..12).toTensor().reshape(1,3,4)

      block: # Single axis with slices
        let a = toSeq(1..12).toTensor().reshape(1,3,1,4)
        let b = a[0, 0..1, 0, 3..0|-1].squeeze(0)

        check: b == [4, 3, 2, 1, 8, 7, 6, 5].toTensor.reshape(2,1,4)

    test "Squeezing a single element tensor":
      # Previously squeezing a single element tensor turned the result into
      # a 0 rank tensor. But we don't actually fully support zero rank tensors!
      # Hence `squeeze` now leaves at least a rank 1 tensor
      block:
        let a = [1].toTensor
        let b = a.squeeze()
        check: a.rank == 1
        check: b.rank == 1
      block:
        let a = [[1]].toTensor
        let b = a.squeeze()
        check: a.rank == 2
        check: b.rank == 1
      block:
        let a = [[[1]]].toTensor
        let b = a.squeeze()
        check: a.rank == 3
        check: b.rank == 1
      block:
        let a = [[[[1]]]].toTensor
        let b = a.squeeze()
        check: a.rank == 4
        check: b.rank == 1

    test "Getting the item out of a single element tensor":
      block:
        let a = [[[[1.5]]]].toTensor
        let value = a.item()
        check value == 1.5
      block:
        let a = [[[[1]]]].toTensor
        let value = a.item(Complex64)
        check value == complex(1.0, 0)
      block:
        let a = [[[[complex[float64](1.0, 1.1)]]]].toTensor
        let value = a.item(Complex32)
        check value == complex[float32](1.0, 1.1)

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

    test "Split tensors":
      let a = toSeq(1..12).toTensor
      check:
        a.split(3, axis = 0) == [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]].mapIt(it.toTensor)
        a.split(4, axis = 0) == [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]].mapIt(it.toTensor)
        a.split(5, axis = 0) == [@[1, 2, 3, 4, 5], @[6, 7, 8, 9, 10], @[11, 12]].mapIt(it.toTensor)

    test "Chunk tensors":
      let a = toSeq(1..12).toTensor
      check:
        a.chunk(3, axis = 0) == [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]].mapIt(it.toTensor)
        a.chunk(4, axis = 0) == [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]].mapIt(it.toTensor)
        a.chunk(5, axis = 0) == [@[1, 2, 3], @[4, 5, 6], @[7, 8], @[9, 10], @[11, 12]].mapIt(it.toTensor)

    test "Roll":
      block: # Rank-1 tensor roll
        let a = arange(5)
        check:
          a.roll(0) == [0, 1, 2, 3, 4].toTensor
          a.roll(1) == [4, 0, 1, 2, 3].toTensor
          a.roll(2) == [3, 4, 0, 1, 2].toTensor
          a.roll(7) == [3, 4, 0, 1, 2].toTensor
          a.roll(-2) == [2, 3, 4, 0, 1].toTensor
          # rolling over an axis is the same as a global roll for rank-1 tensors
          a.roll(2) == a.roll(2, axis=0)

      block: # Rank-2 tensor roll over an axis
        let a = arange(12).reshape(3, 4)
        check:
          a.roll(1, axis = 0) == [[8, 9, 10, 11], [0, 1, 2, 3], [4, 5, 6, 7]].toTensor
          a.roll(-1, axis = 0) == [[4, 5, 6, 7], [8, 9, 10, 11], [0, 1, 2, 3]].toTensor
          a.roll(-4, axis = 0) == a.roll(2, axis = 0)

          a.roll(1, axis = 1) == [[3, 0, 1, 2], [7, 4, 5, 6], [11, 8, 9, 10]].toTensor
          a.roll(-1, axis = 1) == [[1, 2, 3, 0], [5, 6, 7, 4], [9, 10, 11, 8]].toTensor
          a.roll(-5, axis = 1) == a.roll(3, axis = 1)

      block: # Rank-3 tensor roll over an axis
        let a = arange(24).reshape(2, 3, 4)
        check:
          a.roll(1, axis = 0) == [[[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]],
                                  [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]]
                                  .toTensor
          a.roll(-1, axis = 0) == a.roll(1, axis = 0)

          a.roll(1, axis = 1) == [[[8, 9, 10, 11], [0, 1, 2, 3], [4, 5, 6, 7]],
                                  [[20, 21, 22, 23], [12, 13, 14, 15], [16, 17, 18, 19]]]
                                  .toTensor
          a.roll(-1, axis = 1) == [[[4, 5, 6, 7], [8, 9, 10, 11], [0, 1, 2, 3]],
                                   [[16, 17, 18, 19], [20, 21, 22, 23], [12, 13, 14, 15]]]
                                   .toTensor

          a.roll(1, axis = 2) == [[[3, 0, 1, 2], [7, 4, 5, 6], [11, 8, 9, 10]],
                                  [[15, 12, 13, 14], [19, 16, 17, 18], [23, 20, 21, 22]]]
                                  .toTensor
          a.roll(-1, axis = 2) == [[[1, 2, 3, 0], [5, 6, 7, 4], [9, 10, 11, 8]],
                                   [[13, 14, 15, 12], [17, 18, 19, 16], [21, 22, 23, 20]]]
                                   .toTensor
          a.roll(-6, axis = 2) == a.roll(2, axis = 2)

      block: # Combined rolls
        let a = arange(8).reshape(2, 4)
        check:
          a.roll(1, axis=0).roll(2, axis=1) == [[6, 7, 4, 5], [2, 3, 0, 1]].toTensor

      block: # Global (axis-less) roll
        let a = arange(12).reshape(3, 4)
        check:
          a.roll(1) == [[11, 0, 1, 2], [3, 4, 5, 6], [7, 8, 9, 10]].toTensor
          a.roll(-1) == [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 0]].toTensor
          a.roll(-5) == a.roll(7)

    test "Axis permute and move":
      block: # Permute and moveaxis
        let a = arange(6).reshape(1, 2, 3)
        let a_permuted_1 = [[[0, 3], [1, 4], [2, 5]]].toTensor
        let a_permuted_2 = [[[0], [1], [2]], [[3], [4], [5]]].toTensor
        check:
          # Keep dim 0 at 0 and swap dimensions 1 and 2
          a_permuted_1 == a.permute(0, 2, 1)
          a_permuted_1 == a.moveaxis(2, 1)

          # Move dim 0 to 2, which is the same as
          # moving dim 1 to 0 and then moving 2 to 1
          a_permuted_2 == a.permute(1, 2, 0)
          a_permuted_2 == a.moveaxis(1, 0).moveaxis(2, 1)

main()
GC_fullCollect()
