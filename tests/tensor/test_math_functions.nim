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
import std/[unittest, math]
import complex except Complex64, Complex32

proc main() =
  suite "Math functions":
    test "Reciprocal (element-wise 1/x)":
      var a = [1.0, 10, 20, 30].toTensor.reshape(4,1)
      var a_c = [1.0, 10, 20, 30].toTensor.reshape(4,1).asType(Complex[float64])


      check: a.reciprocal == [[1.0],
                              [1.0/10.0],
                              [1.0/20.0],
                              [1.0/30.0]].toTensor
      check: a_c.reciprocal == [[1.0],
                              [1.0/10.0],
                              [1.0/20.0],
                              [1.0/30.0]].toTensor.asType(Complex[float64])

      a.mreciprocal
      a_c.mreciprocal

      check: a == [[1.0],
                  [1.0/10.0],
                  [1.0/20.0],
                  [1.0/30.0]].toTensor
      check: a_c == [[1.0],
                  [1.0/10.0],
                  [1.0/20.0],
                  [1.0/30.0]].toTensor.asType(Complex[float64])

    test "Negate elements (element-wise -x)":
      block: # Out of place
        var a = [1.0, 10, 20, 30].toTensor.reshape(4,1)


        check: a.negate == [[-1.0],
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
      var a_c = [1.0, -10, -20, 30].toTensor.reshape(4,1).asType(Complex[float64])

      check: a.abs == [[1.0],
                        [10.0],
                        [20.0],
                        [30.0]].toTensor
      check: a_c.abs == [[1.0],
                        [10.0],
                        [20.0],
                        [30.0]].toTensor.asType(float64)

      a.mabs

      check: a == [[1.0],
                  [10.0],
                  [20.0],
                  [30.0]].toTensor

    test "Complex Phase":
      var a_c = [[
        complex(0.0, 0.0),
        complex(1.0, 0.0),
        complex(1.0, 1.0),
        complex(0.0, 1.0),
        complex(-1.0, 1.0),
        complex(-1.0, 0.0),
        complex(-1.0, -1.0),
        complex(0.0, -1.0)
      ]].toTensor

      var expected_phases = [[0.0, 0.0, PI/4.0, PI/2.0, 3.0*PI/4.0, PI, -3.0*PI/4.0, -PI/2.0]].toTensor

      check: a_c.phase == expected_phases

    test "1-D convolution":
      block:
        let a = arange(4)
        let b = 2 * ones[int](7) - arange(7)
        let expected = [0, 2, 5, 8, 2, -4, -10, -16, -17, -12].toTensor
        let expected_same = [2, 5, 8, 2, -4, -10, -16].toTensor
        let expected_valid = [8, 2, -4, -10].toTensor

        check: convolve(a, b) == expected
        # Test that input order doesn't matter
        check: convolve(b, a) == expected
        # Test the `same` mode with different input sizes
        check: convolve(a, b, mode=ConvolveMode.same) == expected_same

        let a2 = arange(5)
        let b2 = 2 * ones[int](8) - arange(8)
        let expected_same_a2b = [  5,   8,  10,   0, -10, -20, -25].toTensor
        let expected_same_ab2 = [  2,   5,   8,   2,  -4, -10, -16, -22].toTensor
        check: convolve(a2, b, mode=ConvolveMode.same) == expected_same_a2b
        check: convolve(a, b2, mode=ConvolveMode.same) == expected_same_ab2

        # Test the `valid` mode
        check: convolve(b, a, mode=ConvolveMode.valid) == expected_valid

main()
GC_fullCollect()
