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
import math, unittest

suite "Universal functions":
  test "Common math functions are exported":
    let a = @[@[1.0,2,3],@[4.0,5,6]]
    let b = @[@[7.0, 8],@[9.0, 10],@[11.0, 12]]

    let ta = a.toTensor(Cpu)
    let tb = b.toTensor(Cpu)

    let expected_a = @[@[cos(1'f64),cos(2'f64),cos(3'f64)],@[cos(4'f64),cos(5'f64),cos(6'f64)]]
    let expected_b = @[@[ln(7'f64), ln(8'f64)],@[ln(9'f64), ln(10'f64)],@[ln(11'f64), ln(12'f64)]]

    check: cos(ta) == expected_a.toTensor(Cpu)
    check: ln(tb) == expected_b.toTensor(Cpu)

  test "Creating custom universal functions is supported":
    proc square_plus_one(x: int): int = x ^ 2 + 1
    makeUniversalLocal(square_plus_one)

    let c = @[@[2,4,8],@[3,9,27]]
    let tc = c.toTensor(Cpu)

    let expected_c = @[@[5, 17, 65],@[10, 82, 730]]

    check: square_plus_one(tc) == expected_c.toTensor(Cpu)

  ## MakeUniversal cannot change Tensor[B,T] to Tensor[B,U] for now
  ## fmap must be used instead
  test "Universal functions that change types are supported":
    let d = @[@[2,4,8],@[3,9,27]]
    let e = @[@["2","4","8"],@["3","9","27"]]

    proc stringify(n: int): string = $n
    # makeUniversalLocal(stringify)

    let td = d.toTensor(Cpu)
    let te = e.toTensor(Cpu)

    when compiles (td == te): check: false

    check: td.fmap(stringify) == te
    check: td.fmap(stringify)[0,1] == "4"
    expect(IndexError):
      discard td.fmap(stringify)[1,3]