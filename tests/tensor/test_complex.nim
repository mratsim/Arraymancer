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

proc main() =
  suite "Basic Complex Tensor Operations":
    test "Complex Tensor Creation":
      var re = [1.0, -10, 20].toTensor
      var im = [-300.0, 20, -1].toTensor
      var c = complex(re, im)
      var expected_c = [
        complex(1.0, -300.0),
        complex(-10.0, 20.0),
        complex(20.0, -1.0),
      ].toTensor

      check: c == expected_c

    test "Get the Real and Imaginary Components of a Complex Tensor":
      var c = [
        complex(1.0, -300.0),
        complex(-10.0, 20.0),
        complex(20.0, -1.0),
      ].toTensor
      var re = c.real
      var im = c.imag

      var expected_re = [1.0, -10, 20].toTensor
      var expected_im = [-300.0, 20, -1].toTensor

      check:
        re == expected_re
        im == expected_im

    test "Set the Real and Imaginary Components of a Complex Tensor":
      var c = newTensor[Complex64](3)

      # Set the real components to a single value
      c.real = 3.0
      # Set the imaginary component to a tensor
      c.imag = arange(3).asType(float)

      var expected_c = [
        complex(3.0, 0.0),
        complex(3.0, 1.0),
        complex(3.0, 2.0),
      ].toTensor

      check: c == expected_c

    test "Complex Conjugate":
      var c = [
        complex(1.0, -300.0),
        complex(-10.0, 20.0),
        complex(20.0, -1.0),
      ].toTensor

      var expected_c_conjugate = [
        complex(1.0, 300.0),
        complex(-10.0, -20.0),
        complex(20.0, 1.0),
      ].toTensor

      check: c.conjugate == expected_c_conjugate

main()
GC_fullCollect()
