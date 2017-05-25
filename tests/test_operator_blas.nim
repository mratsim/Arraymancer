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
import unittest, future

suite "BLAS (Basic Linear Algebra Subprograms)":
  test "GEMM - General Matrix to Matrix Multiplication":
    ## TODO: test with slices
    let a = @[@[1.0,2,3],@[4.0,5,6]]
    let b = @[@[7.0, 8],@[9.0, 10],@[11.0, 12]]

    let ta = a.toTensor(Cpu)
    let tb = b.toTensor(Cpu)

    let expected = @[@[58.0,64],@[139.0,154]]
    let t_expected = expected.toTensor(Cpu)

    check: ta * tb == t_expected


  test "GEMM - Bounds checking":
    let c = @[@[1'f32,2,3],@[4'f32,5,6]]
    let tc = c.toTensor(Cpu)

    when compiles(tc * tb): check: false
    expect(IndexError):
       discard tc * tc

  test "GEMV - General Matrix to Vector Multiplication":
    ## TODO: test with slices
    let d_int = @[@[1,-1,2],@[0,-3,1]]
    let e_int = @[2, 1, 0]
    let tde_expected_int = @[1, -3]

    let td_int = d_int.toTensor(Cpu)
    let te_int = e_int.toTensor(Cpu)

    ## TODO integer fallback
    # check: td_int * te_int == tde_expected_int.toTensor(Cpu)

    let d_float = @[@[1.0,-1,2],@[0.0,-3,1]]
    let e_float = @[2.0, 1, 0]

    let td_float = d_float.toTensor(Cpu)
    let te_float = e_float.toTensor(Cpu)

    check: td_float * te_float == tde_expected_int.toTensor(Cpu).fmap(x => x.float64)

  test "GEMM and GEMV with transposed matrices":
    let a = @[@[1.0,2,3],@[4.0,5,6]]
    let ta = a.toTensor(Cpu)
    let b = @[@[7.0, 8],@[9.0, 10],@[11.0, 12]]
    let tb = b.toTensor(Cpu)


    let at = @[@[1.0,4],@[2.0,5],@[3.0,6]]
    let tat = at.toTensor(Cpu)

    let expected = @[@[58.0,64],@[139.0,154]]
    let t_expected = expected.toTensor(Cpu)

    check: transpose(tat) * tb == t_expected

    let bt = @[@[7.0, 9, 11],@[8.0, 10, 12]]
    let tbt = bt.toTensor(Cpu)

    check: ta * transpose(tbt) == t_expected

    check: transpose(tat) * transpose(tbt) == t_expected


    let d = @[@[1.0,-1,2],@[0.0,-3,1]]
    let e = @[2.0, 1, 0]

    let td = d.toTensor(Cpu)
    let te = e.toTensor(Cpu)

    let dt = @[@[1.0,0],@[-1.0,-3],@[2.0,1]]
    let tdt = dt.toTensor(Cpu)

    check: td * te == transpose(tdt) * te

  test "Scalar/dot product":
    ## TODO: test with slices
    let u_int = @[1, 3, -5]
    let v_int = @[4, -2, -1]

    let tu_int = u_int.toTensor(Cpu)
    let tv_int = v_int.toTensor(Cpu)

    check: tu_int .* tv_int == 3


    let u_float = @[1'f64, 3, -5]
    let v_float = @[4'f64, -2, -1]

    let tu_float = u_float.toTensor(Cpu)
    let tv_float = v_float.toTensor(Cpu)

    check: tu_float .* tv_float == 3.0

  test "Multiplication/division by scalar":
    let u_int = @[1, 3, -5]
    let u_expected = @[2, 6, -10]
    let tu_int = u_int.toTensor(Cpu)

    check: 2 * tu_int == u_expected.toTensor(Cpu)
    check: tu_int * 2 == u_expected.toTensor(Cpu)

    let u_float = @[1'f64, 3, -5]
    let tu_float = u_float.toTensor(Cpu)

    let ufl_expected = @[2'f64, 6, -10]
    check: ufl_expected.toTensor(Cpu) / 2 == tu_float

  test "Tensor addition and substraction":
    let u_int = @[1, 3, -5]
    let v_int = @[1, 1, 1]
    let expected_add = @[2, 4, -4]
    let expected_sub = @[0, 2, -6]
    let tu_int = u_int.toTensor(Cpu)
    let tv_int = v_int.toTensor(Cpu)

    check: tu_int + tv_int == expected_add.toTensor(Cpu)
    check: tu_int - tv_int == expected_sub.toTensor(Cpu)

  test "Addition-Substraction - slices":
    let a = @[@[1.0,2,3],@[4.0,5,6], @[7.0,8,9]]
    let ta = a.toTensor(Cpu)
    let ta_t = ta.transpose()

    check: ta[0..1, 0..1] + ta_t[0..1, 0..1] == [[2.0, 6], [6.0, 10]].toTensor(Cpu)
    check: ta[1..2, 1..2] - ta_t[1..2, 1..2] == [[0.0, -2], [2.0, 0]].toTensor(Cpu)

  test "Addition-Substraction - Bounds checking":
    let a = [[1.0,2,3], [4.0,5,6], [7.0,8,9]]
    let ta = a.toTensor(Cpu)
    let ta_t = ta.transpose()

    expect(ValueError):
      discard ta[1..2,1..2] + ta_t

    expect(ValueError):
      discard ta - ta_t[1..2,1..2]