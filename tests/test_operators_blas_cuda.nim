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

import ../src/cuda
import unittest, future

suite "CUDA backend: BLAS (Basic Linear Algebra Subprograms)":
  test "Scalar/dot product":
    let u = @[1'f64, 3, -5].toTensor().cuda()
    let v = @[4'f64, -2, -1].toTensor().cuda()

    check: u .* v == 3.0