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


suite "Cuda init":
  test "Clone function":
    let a = [ 7, 4, 3, 1, 8, 6,
              8, 1, 6, 2, 6, 6,
              2, 0, 4, 3, 2, 0].toTensor.reshape([3,6]).astype(float).cuda

    # Tensor of shape 3x6 of type "int" on backend "Cpu"
    # |7      4       3       1       8       6|
    # |8      1       6       2       6       6|
    # |2      0       4       3       2       0|

    let b = a[2..0|-2, 2..0|-2].clone

    check: b.cpu == [ [4.0, 2.0],
                      [3.0, 7.0]].toTensor
