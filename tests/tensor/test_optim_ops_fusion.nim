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


suite "Testing fusion operations":
  test "Multiply and add":
    let
      A = @[4.0].toTensor.reshape(1, 1)
      x = @[3.0].toTensor
      b = zeros[float](1)
    for _ in 0..4:
      check ((A * x) + b)[0] == 12.0
      check b[0] == 0.0
