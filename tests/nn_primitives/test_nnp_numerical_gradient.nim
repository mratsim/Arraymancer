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
import unittest

proc main() =
  suite "Test the numerical gradient proc":
    test "Numerical gradient":
      proc f(x: float): float = x*x + x + 1.0
      check: numerical_gradient(2.0, f).relative_error(5.0) < 1e-8

      proc g(t: Tensor[float]): float =
        let x = t[0]
        let y = t[1]
        x*x + y*y + x*y + x + y + 1.0
      let input = [2.0, 3.0].toTensor()
      let grad = [8.0, 9.0].toTensor()
      check: numerical_gradient(input, g).mean_relative_error(grad) < 1e-8

main()
GC_fullCollect()
