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
import unittest, math
import complex except Complex64, Complex32


suite "Testing miscellaneous data functions":
  test "Copy data from source":
    let a = [[1,2],[3,4]].toTensor.reshape(2,2)

    var b = ones[int](4,1)

    b.copyFrom(a)

    check: b == [[1],[2], [3], [4]].toTensor
    block:
      let a = [[1,2],[3,4]].toTensor.reshape(2,2).astype(Complex[float64])
      var b = ones[Complex[float64]](4,1)
      b.copyFrom(a)
      check: b == [[1],[2], [3], [4]].toTensor.astype(Complex[float64])
