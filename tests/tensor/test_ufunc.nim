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
import math, unittest, complex
import complex except Complex64, Complex32

suite "Universal functions":
  test "As type with slicing":
    let a = [1, 2, 3, 4].toTensor()
    let b = a[1..2].astype(float)
    check b == [2.0'f64,3.0'f64].toTensor()

    let c = a[1..2].astype(Complex[float64])
    check c == [complex64(2.0'f64,0'f64), complex64(3.0'f64,0'f64)].toTensor()

  test "Common math functions are exported":
    let a = @[@[1.0,2,3],@[4.0,5,6]]
    let b = @[@[7.0, 8],@[9.0, 10],@[11.0, 12]]
    let c = @[@[complex64(1.0,0.0),complex64(2.0,0.0),complex64(3.0,0.0)],
              @[complex64(4.0,0.0),complex64(5.0,0.0),complex64(6.0,0.0)]]

    let ta = a.toTensor()
    let tb = b.toTensor()
    let tc = c.toTensor()

    let expected_a = @[@[cos(1'f64),cos(2'f64),cos(3'f64)],@[cos(4'f64),cos(5'f64),cos(6'f64)]]
    let expected_b = @[@[ln(7'f64), ln(8'f64)],@[ln(9'f64), ln(10'f64)],@[ln(11'f64), ln(12'f64)]]
    let expected_c = @[@[cos(complex64(1'f64,0.0)),cos(complex64(2'f64,0.0)),cos(complex64(3'f64,0.0))],
                       @[cos(complex64(4'f64,0.0)),cos(complex64(5'f64,0.0)),cos(complex64(6'f64,0.0))]]

    check: mean_absolute_error(cos(ta), expected_a.toTensor()) <= 1e-15 # We have slight precision loss with OpenMP
    check: abs(mean_absolute_error(cos(tc), expected_c.toTensor())) <= 1e-15
    check: ln(tb) == expected_b.toTensor()

  test "Creating custom universal functions is supported":
    proc square_plus_one(x: int): int = x ^ 2 + 1
    proc square_plus_one(x: Complex[float64]): Complex[float64] = x.pow(2) + 1
    makeUniversalLocal(square_plus_one)

    let c = @[@[2,4,8],@[3,9,27]]
    let tc = c.toTensor()
    let tcc = tc.astype(Complex[float64])

    let expected_c = @[@[5, 17, 65],@[10, 82, 730]]
    let expected_cc = @[@[complex64(5'f64,0.0), complex64(17'f64,0.0), complex64(65'f64,0.0)],
                        @[complex64(10'f64,0.0), complex64(82'f64,0.0), complex64(730'f64,0.0)]]

    check: square_plus_one(tc) == expected_c.toTensor()
    check: square_plus_one(tcc) == expected_cc.toTensor()

  ## MakeUniversal cannot change Tensor[B,T] to Tensor[B,U] for now
  ## map must be used instead
  test "Universal functions that change types are supported":
    let d = @[@[2,4,8],@[3,9,27]]
    let e = @[@["2","4","8"],@["3","9","27"]]

    proc stringify(n: int): string = $n
    # makeUniversalLocal(stringify)

    let td = d.toTensor()
    let te = e.toTensor()

    when compiles(td == te): check: false

    check: td.map(stringify) == te
    check: td.map(stringify)[0,1] == "4"

    when compileOption("boundChecks") and not defined(openmp):
      expect(IndexError):
        discard td.map(stringify)[1,3]
    else:
      echo "Bound-checking is disabled or OpenMP is used. The incorrect seq shape test has been skipped."


  test "Abs":
    let a = [-2,-1,0,1,2].toTensor()
    check abs(a) == [2,1,0,1,2].toTensor()
    let b = [-2.0,-1,0,1,2].toTensor()
    check abs(b) == [2.0,1,0,1,2].toTensor()
    # abs complex -> float
    let c = [-2.0,-1,0,1,2].toTensor().astype(Complex[float64])
    check abs(c) == [2.0,1,0,1,2].toTensor().astype(float64)
