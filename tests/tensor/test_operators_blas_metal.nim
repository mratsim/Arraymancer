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
import std / [unittest, sugar]

suite "Metal BLAS backend (Basic Linear Algebra Subprograms)":
  test "GEMM - General Matrix to Matrix Multiplication":
    let a = [[1.0, 2, 3],
             [4.0, 5, 6]].toTensor().asType(float32).toMetal()

    let b = [[7.0, 8],
             [9.0, 10],
             [11.0, 12]].toTensor().asType(float32).toMetal()

    let expected = [[58.0, 64],
                    [139.0, 154]].toTensor().asType(float32)

    let result = (a * b).toCpu()
    check: result == expected

  test "Matrix-Vector multiplication":
    let a = [[1.0, 2.0, 3.0],
             [4.0, 5.0, 6.0]].toTensor().asType(float32).toMetal()
    let x = [1.0, 2.0, 3.0].toTensor().asType(float32).toMetal()

    let expected = [14.0, 32.0].toTensor().asType(float32)

    let result = (a * x).toCpu()
    check: result == expected

  test "Element-wise addition":
    let a = [[1.0, 2.0], [3.0, 4.0]].toTensor().asType(float32).toMetal()
    let b = [[5.0, 6.0], [7.0, 8.0]].toTensor().asType(float32).toMetal()

    let expected = [[6.0, 8.0], [10.0, 12.0]].toTensor().asType(float32)

    let result = (a + b).toCpu()
    check: result == expected

  test "Element-wise subtraction":
    let a = [[5.0, 6.0], [7.0, 8.0]].toTensor().asType(float32).toMetal()
    let b = [[1.0, 2.0], [3.0, 4.0]].toTensor().asType(float32).toMetal()

    let expected = [[4.0, 4.0], [4.0, 4.0]].toTensor().asType(float32)

    let result = (a - b).toCpu()
    check: result == expected

  test "Scalar multiplication":
    let a = [[1.0, 2.0], [3.0, 4.0]].toTensor().asType(float32).toMetal()

    let expected = [[2.0, 4.0], [6.0, 8.0]].toTensor().asType(float32)

    let result = (2.0'f32 * a).toCpu()
    check: result == expected

  test "Dot product":
    let a = [1.0, 2.0, 3.0, 4.0].toTensor().asType(float32).toMetal()
    let b = [5.0, 6.0, 7.0, 8.0].toTensor().asType(float32).toMetal()

    let result = dot(a, b)
    check: result == 70.0'f32

  test "In-place addition":
    var a = [[1.0, 2.0], [3.0, 4.0]].toTensor().asType(float32).toMetal()
    let b = [[5.0, 6.0], [7.0, 8.0]].toTensor().asType(float32).toMetal()

    let expected = [[6.0, 8.0], [10.0, 12.0]].toTensor().asType(float32)

    a += b
    check: a.toCpu() == expected

  test "Large matrix multiplication":
    # Test with matrices above the GEMM threshold
    let a = randomTensor([200, 200], 1.0'f32).toMetal()
    let b = randomTensor([200, 200], 1.0'f32).toMetal()

    # Just verify it doesn't crash and returns correct shape
    let result = a * b
    check: result.shape == [200, 200]
