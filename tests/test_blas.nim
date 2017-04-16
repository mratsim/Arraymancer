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

import ../arraymancer
import unittest

suite "BLAS (Basic Linear Algebra Subprograms)":
    test "GEMM - General Matrix to Matrix Multiplication":
        let a = @[@[1.0,2,3],@[4.0,5,6]]
        let b = @[@[7.0, 8],@[9.0, 10],@[11.0, 12]]

        let ta = fromSeq(a,float64,Backend.Cpu)
        let tb = fromSeq(b,float64,Backend.Cpu)

        let expected = @[@[58.0,64],@[139.0,154]]
        let t_expected = fromSeq(expected, float64,Backend.Cpu)

        check: ta * tb == t_expected