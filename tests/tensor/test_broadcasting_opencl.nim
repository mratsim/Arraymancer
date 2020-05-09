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
import unittest, sugar, sequtils

suite "OpenCL: Shapeshifting - broadcasting and non linear algebra elementwise operations":
  test "Tensor element-wise multiplication (Hadamard product) and division":
    block:
      let u = @[-4, 0, 9].toTensor().astype(float32).opencl
      let v = @[2, 10, 3].toTensor().astype(float32).opencl
      let expected_mul = @[-8, 0, 27].toTensor().astype(float32)
      let expected_div = @[-2, 0, 3].toTensor().astype(float32)

      check: (u *. v).cpu == expected_mul
      check: (u /. v).cpu == expected_div

    block:
      let u = @[1.0, 8.0, -3.0].toTensor().astype(float32).opencl
      let v = @[4.0, 2.0, 10.0].toTensor().astype(float32).opencl
      let expected_mul = @[4.0, 16.0, -30.0].toTensor().astype(float32)
      let expected_div = @[0.25, 4.0, -0.3].toTensor().astype(float32)

      check: (u *. v).cpu == expected_mul
      check: (u /. v).cpu == expected_div

  test "Implicit tensor-tensor broadcasting - basic operations +., -., *., ./, .^":
    block: # Addition
      let a = [0, 10, 20, 30].toTensor().reshape(4,1).astype(float32).opencl
      let b = [0, 1, 2].toTensor().reshape(1,3).astype(float32).opencl

      check: (a +. b).cpu == [[0, 1, 2],
                              [10, 11, 12],
                              [20, 21, 22],
                              [30, 31, 32]].toTensor.astype(float32)

    block: # Substraction
      let a = [0, 10, 20, 30].toTensor().reshape(4,1).astype(float32).opencl
      let b = [0, 1, 2].toTensor().reshape(1,3).astype(float32).opencl

      check: (a -. b).cpu == [[0, -1, -2],
                              [10, 9, 8],
                              [20, 19, 18],
                              [30, 29, 28]].toTensor.astype(float32)

    block: # Multiplication
      let a = [0, 10, 20, 30].toTensor().reshape(4,1).astype(float32).opencl
      let b = [0, 1, 2].toTensor().reshape(1,3).astype(float32).opencl

      check: (a *. b).cpu == [[0, 0, 0],
                              [0, 10, 20],
                              [0, 20, 40],
                              [0, 30, 60]].toTensor.astype(float32)

    block: # Float division
      let a = [100.0, 10, 20, 30].toTensor().reshape(4,1).opencl
      let b = [2.0, 5, 10].toTensor().reshape(1,3).opencl

      check: (a /. b).cpu == [[50.0, 20, 10],
                              [5.0, 2, 1],
                              [10.0, 4, 2],
                              [15.0, 6, 3]].toTensor
