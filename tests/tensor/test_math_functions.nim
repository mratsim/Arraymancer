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

    test "Sign functions":
      var a = [-5.3, 42.0, -0.0, 0.01, 10.7, -0.001, 0.9, -125.3].toTensor
      let expected_signs = [-1, 1, 0, 1, 1, -1, 1, -1].toTensor()
      check: a.sgn() == expected_signs
      when (NimMajor, NimMinor, NimPatch) >= (1, 6, 0):
        let new_signs = arange(-4.0, 4.0)
        a.mcopySign(new_signs)
        let expected = [-5.3, -42.0, -0.0, -0.01, 10.7, 0.001, 0.9, 125.3].toTensor
        check: a == expected

    test "Modulo functions":
      var a = arange(-70.7, 50.0, 34.7)
      let expected_floormod_ts = [1.3, -0.0, 1.7, 0.4].toTensor()
      let expected_floormod_st = [-67.7, -33.0, -0.9, 3.0].toTensor()
      check: expected_floormod_ts.mean_absolute_error(a.floorMod(3.0)) < 1e-9
      check: expected_floormod_st.mean_absolute_error(floorMod(3.0, a)) < 1e-9

    test "min-max":
      var a = arange(-70, 50, 34)
      var b = arange(53, -73, -34)
      let expected_min = [-70, -36, -15, -49].toTensor()
      let expected_max = [53, 19, -2, 32].toTensor()
      check expected_min == min(a, b)
      check expected_max == max(a, b)

      # N-element versions
      let c = [100, -500, -100, 500].toTensor()
      let expected_n_min = [-70, -500, -100, -49].toTensor()
      let expected_n_max = [100, 19, -2, 500].toTensor()
      check expected_n_min == min(a, b, c)
      check expected_n_max == max(a, b, c)

      # In-place versions
      var d = a.clone()
      d.mmax(b)
      check expected_max == d
      d.mmax(b, c)
      check expected_n_max == d
      d = a.clone()
      d.mmin(b)
      check expected_min == d
      d.mmin(b, c)
      check expected_n_min == d

    test "isNaN & classify":
      var a = [0.0, -0.0, 1.0/0.0, -1.0/0.0, 0.0/0.0].toTensor
      let expected_isNaN = [false, false, false, false, true].toTensor()
      let expected_classification = [fcZero, fcNegZero, fcInf, fcNegInf, fcNaN].toTensor()
      check: expected_isNaN == a.isNaN
      check: expected_classification == a.classify

    test "1-D convolution":
      block:
        let a = arange(4)
        let b = 2 * ones[int](7) - arange(7)
        let expected_full = [0, 2, 5, 8, 2, -4, -10, -16, -17, -12].toTensor
        let expected_same = [2, 5, 8, 2, -4, -10, -16].toTensor
        let expected_valid = [8, 2, -4, -10].toTensor

        # Test all the convolution modes
        check: convolve(a, b, mode=ConvolveMode.full) == expected_full
        check: convolve(a, b, mode=ConvolveMode.same) == expected_same
        check: convolve(a, b, mode=ConvolveMode.valid) == expected_valid

        # Test that the default convolution mode is `full`
        check: convolve(a, b) == expected_full

        # Test that input order doesn't matter
        check: convolve(b, a) == expected_full

        # Test the `same` mode with different input sizes
        let a2 = arange(5)
        let b2 = 2 * ones[int](8) - arange(8)
        let expected_same_a2b = [5, 8, 10, 0, -10, -20, -25].toTensor
        let expected_same_ab2 = [2, 5, 8, 2, -4, -10, -16, -22].toTensor

        check: convolve(a2, b, mode=ConvolveMode.same) == expected_same_a2b
        check: convolve(a, b2, mode=ConvolveMode.same) == expected_same_ab2

    test "1-D correlation":
      block:
        let a = [2, 8, -8, -6, 4].toTensor
        let b = [-7, -7, 6, 0, 6, -7, 5, -6, 2].toTensor
        let expected_full = [4, 4, -54, 62, -40, 50, 26, -30, -94, -36, 122, 14, -28].toTensor
        let expected_same = [-54, 62, -40, 50, 26, -30, -94, -36, 122].toTensor
        let expected_valid = [-40, 50, 26, -30, -94].toTensor

        # Test all the correlation modes
        check: correlate(a, b, mode=ConvolveMode.full) == expected_full
        check: correlate(a, b, mode=ConvolveMode.same) == expected_same
        check: correlate(a, b, mode=ConvolveMode.valid) == expected_valid

        # Test that the default correlation mode is `valid`
        check: correlate(a, b) == expected_valid

      block: # Complex Tensor correlation
        let ca = complex(
          [6.3, 7.1, -6.0, 1.7].toTensor,
          [-8.1, -9.2, 0.0, 3.5].toTensor)
        let cb = complex(
          [-3.9, 1.6, -1.6].toTensor,
          [1.0, 5.2, 2.1].toTensor)
        let expected_full = complex(
          [-27.09, -62.72, -59.55, -41.86, 44.32, -3.13].toTensor,
          [-0.27, -45.91, -13.75, 50.81, 2.76, -15.35].toTensor)
        let expected_same = expected_full[1..^2]
        let expected_valid = expected_full[2..^3]

        # Test all the correlation modes
        check: expected_full.mean_absolute_error(correlate(ca, cb, mode=ConvolveMode.full)) < 1e-9
        check: expected_same.mean_absolute_error(correlate(ca, cb, mode=ConvolveMode.same)) < 1e-9
        check: expected_valid.mean_absolute_error(correlate(ca, cb, mode=ConvolveMode.valid)) < 1e-9

        # Test that the default correlation mode is `valid`
        check: expected_valid.mean_absolute_error(correlate(ca, cb)) < 1e-9

main()
GC_fullCollect()
