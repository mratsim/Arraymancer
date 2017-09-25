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

suite "Shapeshifting - broadcasting and non linear algebra elementwise operations":
  test "Tensor element-wise multiplication (Hadamard product) and division":
    let u_int = @[-4, 0, 9].toTensor()
    let v_int = @[2, 10, 3].toTensor()
    let expected_mul_int = @[-8, 0, 27].toTensor()
    let expected_div_int = @[-2, 0, 3].toTensor()

    check: u_int .* v_int == expected_mul_int
    check: u_int ./ v_int == expected_div_int

    let u_float = @[1.0, 8.0, -3.0].toTensor()
    let v_float = @[4.0, 2.0, 10.0].toTensor()
    let expected_mul_float = @[4.0, 16.0, -30.0].toTensor()
    let expected_div_float = @[0.25, 4.0, -0.3].toTensor()

    check: u_float .* v_float == expected_mul_float
    check: u_float ./ v_float == expected_div_float

  test "Explicit broadcasting":
    block:
      let a = 2.bc([3,2])
      check a == [[2,2],
                  [2,2],
                  [2,2]].toTensor()

    block:
      let a = toSeq(1..2).toTensor().reshape(1,2)
      let b = a.bc([2,2])
      check b == [[1,2],
                  [1,2]].toTensor()

    block:
      let a = toSeq(1..2).toTensor().reshape(2,1)
      let b = a.bc([2,2])
      check b == [[1,1],
                  [2,2]].toTensor()

  test "Implicit broadcasting - basic operations .+, .-, .*, ./":
    block: # Addition
      let a = [0, 10, 20, 30].toTensor().reshape(4,1)
      let b = [0, 1, 2].toTensor().reshape(1,3)

      check: a .+ b == [[0, 1, 2],
                        [10, 11, 12],
                        [20, 21, 22],
                        [30, 31, 32]].toTensor

    block: # Substraction
      let a = [0, 10, 20, 30].toTensor().reshape(4,1)
      let b = [0, 1, 2].toTensor().reshape(1,3)

      check: a .- b == [[0, -1, -2],
                        [10, 9, 8],
                        [20, 19, 18],
                        [30, 29, 28]].toTensor

    block: # Multiplication
      let a = [0, 10, 20, 30].toTensor().reshape(4,1)
      let b = [0, 1, 2].toTensor().reshape(1,3)

      check: a .* b == [[0, 0, 0],
                        [0, 10, 20],
                        [0, 20, 40],
                        [0, 30, 60]].toTensor

    block: # Integer division
      let a = [100, 10, 20, 30].toTensor().reshape(4,1)
      let b = [2, 5, 10].toTensor().reshape(1,3)

      check: a ./ b == [[50, 20, 10],
                        [5, 2, 1],
                        [10, 4, 2],
                        [15, 6, 3]].toTensor

    block: # Float division
      let a = [100.0, 10, 20, 30].toTensor().reshape(4,1)
      let b = [2.0, 5, 10].toTensor().reshape(1,3)

      check: a ./ b == [[50.0, 20, 10],
                        [5.0, 2, 1],
                        [10.0, 4, 2],
                        [15.0, 6, 3]].toTensor

  test "Implicit broadcasting - basic in-place operations .+=, .-=, .*=, ./=":
    block: # Addition
      # Note: We can't broadcast the lhs with in-place operations
      var a = [0, 10, 20, 30].toTensor().reshape(4,1).bc([4,3]).asContiguous
      let b = [0, 1, 2].toTensor().reshape(1,3)

      a .+= b
      check: a == [[0, 1, 2],
                  [10, 11, 12],
                  [20, 21, 22],
                  [30, 31, 32]].toTensor

    block: # Substraction
      # Note: We can't broadcast the lhs with in-place operations
      var a = [0, 10, 20, 30].toTensor().reshape(4,1).bc([4,3]).asContiguous
      let b = [0, 1, 2].toTensor().reshape(1,3)

      a .-= b
      check: a  == [[0, -1, -2],
                    [10, 9, 8],
                    [20, 19, 18],
                    [30, 29, 28]].toTensor

    block: # Multiplication
      # Note: We can't broadcast the lhs with in-place operations
      var a = [0, 10, 20, 30].toTensor().reshape(4,1).bc([4,3]).asContiguous
      let b = [0, 1, 2].toTensor().reshape(1,3)

      a .*= b
      check: a == [[0, 0, 0],
                  [0, 10, 20],
                  [0, 20, 40],
                  [0, 30, 60]].toTensor

    block: # Integer division
      # Note: We can't broadcast the lhs with in-place operations
      var a = [100, 10, 20, 30].toTensor().reshape(4,1).bc([4,3]).asContiguous
      let b = [2, 5, 10].toTensor().reshape(1,3)

      a ./= b
      check: a == [[50, 20, 10],
                  [5, 2, 1],
                  [10, 4, 2],
                  [15, 6, 3]].toTensor

    block: # Float division
      # Note: We can't broadcast the lhs with in-place operations
      var a = [100.0, 10, 20, 30].toTensor().reshape(4,1).bc([4,3]).asContiguous
      let b = [2.0, 5, 10].toTensor().reshape(1,3)

      a ./= b
      check: a == [[50.0, 20, 10],
                  [5.0, 2, 1],
                  [10.0, 4, 2],
                  [15.0, 6, 3]].toTensor

  test "Implicit broadcasting - basic operations .+=, .-= with scalar":
    block: # Addition
      var a = [0, 10, 20, 30].toTensor().reshape(4,1)

      a .+= 100
      check: a == [[100],
                  [110],
                  [120],
                  [130]].toTensor

    block: # Substraction
      var a = [0, 10, 20, 30].toTensor().reshape(4,1)

      a .-= 100
      check: a  == [[-100],
                    [-90],
                    [-80],
                    [-70]].toTensor

  test "Implicit broadcasting - Sigmoid 1 ./ (1 .+ exp(-x)":
    block:
      proc sigmoid[T: SomeReal](t: Tensor[T]): Tensor[T]=
        1.T ./ (1.T .+ exp(-t))

      let a = newTensor[float32]([2,2])
      check: sigmoid(a) == [[0.5'f32, 0.5],[0.5'f32, 0.5]].toTensor