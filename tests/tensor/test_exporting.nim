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

suite "Exporting":
  # TODO Deprecated: toRawSeq cannot be re-implemented in a backward compatible way to v0.6.0
  test "Raw sequence exporting":
    let t = toSeq(1..6).toTensor().reshape(2, 3)
    check t.toRawSeq == toSeq(1 .. 6)

  test "Nested sequence exporting":
    block:
      let s = @[@[1, 2, 3], @[4, 5, 6]]
      check s.toTensor().toSeq2D() == s
    block:
      let s = @[@[complex64(1'f64,0.0), complex64(2'f64,0.0), complex64(3'f64,0.0)],
                @[complex64(4'f64,0.0), complex64(5'f64,0.0), complex64(6'f64,0.0)]]
      check s.toTensor().toSeq2D() == s
    block:
      let t = toSeq(1..24).toTensor().reshape(2, 3, 4)
      check t.toSeq3D().len == 2
      check t.toSeq3D()[0].len == 3
      check t.toSeq3D()[0][0].len == 4
    block:
      let t = toSeq(1..120).toTensor().reshape(2, 3, 4, 5)
      check t.toSeq4D().len == 2
      check t.toSeq4D()[0].len == 3
      check t.toSeq4D()[0][0].len == 4
      check t.toSeq4D()[0][0][0].len == 5
    block:
      let t = toSeq(1..720).toTensor().reshape(2, 3, 4, 5, 6)
      check t.toSeq5D().len == 2
      check t.toSeq5D()[0].len == 3
      check t.toSeq5D()[0][0].len == 4
      check t.toSeq5D()[0][0][0].len == 5
      check t.toSeq5D()[0][0][0][0].len == 6
