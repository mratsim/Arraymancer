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
import unittest, sugar, math

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

  test "Absolute value":
    var a = [1.0, -10, -20, 30].toTensor.reshape(4,1)

    check: a.abs  == [[1.0],
                      [10.0],
                      [20.0],
                      [30.0]].toTensor

    a.mabs

    check: a == [[1.0],
                [10.0],
                [20.0],
                [30.0]].toTensor
