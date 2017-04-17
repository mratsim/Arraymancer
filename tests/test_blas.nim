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
        let a = @[@[1.0,2,3],@[4.0,5,6]]
        let b = @[@[7.0, 8],@[9.0, 10],@[11.0, 12]]

        let ta = fromSeq(a,float64,Backend.Cpu)
        let tb = fromSeq(b,float64,Backend.Cpu)

        let expected = @[@[58.0,64],@[139.0,154]]
        let t_expected = fromSeq(expected, float64,Backend.Cpu)

        check: ta * tb == t_expected


    test "GEMM - Bounds checking":
        let c = @[@[1'f32,2,3],@[4'f32,5,6]]
        let tc = fromSeq(c,float32,Backend.Cpu)

        when compiles(tc * tb): check: false
        expect(IndexError):
           discard tc * tc

    test "GEMV - General Matrix to Vector Multiplication":
        let d_int = @[@[1,-1,2],@[0,-3,1]]
        let e_int = @[2, 1, 0]
        let tde_expected_int = @[1, -3]

        let td_int = fromSeq(d_int, int, Backend.Cpu)
        let te_int = fromSeq(e_int, int, Backend.Cpu)

        ## TODO integer fallback
        # check: td_int * te_int == fromSeq(tde_expected_int, int, Backend.Cpu)

        let d_float = @[@[1.0,-1,2],@[0.0,-3,1]]
        let e_float = @[2.0, 1, 0]

        let td_float = fromSeq(d_float, float64, Backend.Cpu)
        let te_float = fromSeq(e_float, float64, Backend.Cpu)

        check: td_float * te_float == fromSeq(tde_expected_int, int, Backend.Cpu).fmap(x => x.float64)

    test "GEMM and GEMV with transposed matrices":
        let a = @[@[1.0,2,3],@[4.0,5,6]]
        let ta = fromSeq(a,float64,Backend.Cpu)
        let b = @[@[7.0, 8],@[9.0, 10],@[11.0, 12]]
        let tb = fromSeq(b,float64,Backend.Cpu)


        let at = @[@[1.0,4],@[2.0,5],@[3.0,6]]
        let tat = fromSeq(at,float64,Backend.Cpu)

        let expected = @[@[58.0,64],@[139.0,154]]
        let t_expected = fromSeq(expected, float64,Backend.Cpu)

        check: transpose(tat) * tb == t_expected

        let bt = @[@[7.0, 9, 11],@[8.0, 10, 12]]
        let tbt = fromSeq(bt,float64,Backend.Cpu)

        check: ta * transpose(tbt) == t_expected

        check: transpose(tat) * transpose(tbt) == t_expected


        let d = @[@[1.0,-1,2],@[0.0,-3,1]]
        let e = @[2.0, 1, 0]

        let td = fromSeq(d, float64, Backend.Cpu)
        let te = fromSeq(e, float64, Backend.Cpu)

        let dt = @[@[1.0,0],@[-1.0,-3],@[2.0,1]]
        let tdt = fromSeq(dt, float64, Backend.Cpu)

        check: td * te == transpose(tdt) * te

    test "Scalar/dot product":
        let u_int = @[1, 3, -5]
        let v_int = @[4, -2, -1]

        let tu_int = fromSeq(u_int,int,Backend.Cpu)
        let tv_int = fromSeq(u_int,int,Backend.Cpu)

        check: tu_int .* tv_int == 35


        let u_float = @[1'f64, 3, -5]
        let v_float = @[4'f64, -2, -1]

        let tu_float = fromSeq(u_float,float64,Backend.Cpu)
        let tv_float = fromSeq(u_float,float64,Backend.Cpu)

        check: tu_float .* tv_float == 35.0

    test "Multiplication/division by scalar":
        let u_int = @[1, 3, -5]
        let u_expected = @[2, 6, -10]
        let tu_int = fromSeq(u_int,int,Backend.Cpu)

        check: 2 * tu_int == fromSeq(u_expected,int,Backend.Cpu)
        check: tu_int * 2 == fromSeq(u_expected,int,Backend.Cpu)

        let u_float = @[1'f64, 3, -5]
        let tu_float = fromSeq(u_float,float64,Backend.Cpu)

        let ufl_expected = @[2'f64, 6, -10]
        check: fromSeq(ufl_expected,float64,Backend.Cpu) / 2 == tu_float