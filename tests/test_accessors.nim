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


suite "Accessing and setting tensor values":
    test "Accessing and setting a single value":
        var a = newTensor(@[2,3,4], int, Backend.Cpu)
        a[1,2,2] = 122
        check: a[1,2,2] == 122
        echo a

        var b = newTensor(@[3,4], int, Backend.Cpu)
        b[1,2] = 12
        check: b[1,2] == 12
        b[0,0] = 999
        check: b[0,0] == 999
        b[2,3] = 111
        check: b[2,3] == 111
        b[2,0] = 555
        # echo b

#    test "Out of bounds checking":
#        # Cannot test properly "when compiles assignation"
#        var a = newTensor(@[2,3,4], int, Backend.Cpu)
#        when compiles(a[2,0,0] = 200): false
#        var b = newTensor(@[3,4], int, Backend.Cpu)
#        when compiles(b[3,4] = 999): false
#        echo b