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

# Please compile with -d:metal switch
import ../../src/arraymancer
import std / [unittest, sugar, sequtils]

suite "Metal: Shapeshifting - broadcasting and non linear algebra elementwise operations":
  test "Tensor element-wise multiplication (Hadamard product) and division":
    block:
      let u = @[-4, 0, 9].toTensor().asType(float32).toMetal()
      let v = @[2, 10, 3].toTensor().asType(float32).toMetal()
      let expected_mul = @[-8, 0, 27].toTensor().asType(float32)
      let expected_div = @[-2, 0, 3].toTensor().asType(float32)

      check: (u *. v).toCpu() == expected_mul
      check: (u /. v).toCpu() == expected_div

    block:
      let u = @[1.0, 8.0, -3.0].toTensor().asType(float32).toMetal()
      let v = @[4.0, 2.0, 10.0].toTensor().asType(float32).toMetal()
      let expected_mul = @[4.0, 16.0, -30.0].toTensor().asType(float32)
      let expected_div = @[0.25, 4.0, -0.3].toTensor().asType(float32)

      check: (u *. v).toCpu() == expected_mul
      check: (u /. v).toCpu() == expected_div

  test "Tensor element-wise in-place multiplication (Hadamard product) and division":
    block:
      var u = @[-4.0, 0.0, 9.0].toTensor().asType(float32).toMetal()
      let v = @[2.0, 10.0, 3.0].toTensor().asType(float32).toMetal()
      let expected_mul = @[-8.0, 0.0, 27.0].toTensor().asType(float32)

      u *.= v
      check: u.toCpu() == expected_mul

    block:
      var u = @[100.0, 10.0, 30.0].toTensor().asType(float32).toMetal()
      let v = @[2.0, 5.0, 10.0].toTensor().asType(float32).toMetal()
      let expected_div = @[50.0, 2.0, 3.0].toTensor().asType(float32)

      u /.= v
      check: u.toCpu() == expected_div
