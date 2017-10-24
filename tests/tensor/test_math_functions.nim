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

import ../../src/arraymancer
import unittest, future, math

suite "Math functions":
  test "Reciprocal (element-wise 1/x)":
    var a = [1.0, 10, 20, 30].toTensor.reshape(4,1)


    check: a.reciprocal  == [[1.0],
                            [1.0/10.0],
                            [1.0/20.0],
                            [1.0/30.0]].toTensor

    a.mreciprocal

    check: a == [[1.0],
                [1.0/10.0],
                [1.0/20.0],
                [1.0/30.0]].toTensor

  test "Negate elements (element-wise -x)":
    block: # Out of place
      var a = [1.0, 10, 20, 30].toTensor.reshape(4,1)


      check: a.negate  == [[-1.0],
                          [-10.0],
                          [-20.0],
                          [-30.0]].toTensor

      a.mnegate

      check: a == [[-1.0],
                  [-10.0],
                  [-20.0],
                  [-30.0]].toTensor

  test "Clamp":
    var a = [-5,2,3,5,10,0,1,-1].toTensor()
    let target = [-2,2,2,2,2,0,1,-1].toTensor()
    check: a.clamp(-2,2) == target
    a.mclamp(-2,2)
    check: a == target

  test "Numerical gradient":
    proc f(x: float): float = x*x + x + 1.0
    check: relative_error(numerical_gradient(2.0, f), 5.0) < 1e-8

    proc g(t: Tensor[float]): float =
      let x = t[0]
      let y = t[1]
      x*x + y*y + x*y + x + y + 1.0
    let input = [2.0, 3.0].toTensor()
    let grad = [8.0, 9.0].toTensor()
    check: mean_relative_error(numerical_gradient(input, g), grad) < 1e-8

  test "Mean absolute error":
    var y_true = [0.9, 0.2, 0.1, 0.4, 0.9].toTensor()
    var y =      [1.0, 0.0, 0.0, 1.0, 1.0].toTensor()
    check: absolute_error(y_true, y).sum() == 1.1
    check: mean_absolute_error(y_true, y) == (1.1 / 5.0)

  test "Mean squared error":
    var y_true = [0.9, 0.2, 0.1, 0.4, 0.9].toTensor()
    var y =      [1.0, 0.0, 0.0, 1.0, 1.0].toTensor()
    check: squared_error(y_true, y).sum() == 0.43
    check: mean_squared_error(y_true, y) == (0.43 / 5.0)

  test "Relative error":
    var y_true = [0.0,  0.0, -1.0, 1e-8, 1e-8].toTensor()
    var y =      [0.0, -1.0,  0.0, 0.0,  1e-7].toTensor()
    check: relative_error(y_true, y) == [0.0, 1.0, 1.0, 1.0, 0.9].toTensor()
    check: mean_relative_error(y_true, y) == 0.78