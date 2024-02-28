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
import unittest, sequtils, math
import complex except Complex64, Complex32

proc main() =
  suite "Shapeshifting - broadcasting and non linear algebra elementwise operations":
    test "Tensor element-wise multiplication (Hadamard product) and division":
      let u_int = @[-4, 0, 9].toTensor()
      let v_int = @[2, 10, 3].toTensor()
      let expected_mul_int = @[-8, 0, 27].toTensor()
      let expected_div_int = @[-2, 0, 3].toTensor()

      check: u_int *. v_int == expected_mul_int
      check: u_int /. v_int == expected_div_int

      let u_float = @[1.0, 8.0, -3.0].toTensor()
      let v_float = @[4.0, 2.0, 10.0].toTensor()
      let expected_mul_float = @[4.0, 16.0, -30.0].toTensor()
      let expected_div_float = @[0.25, 4.0, -0.3].toTensor()

      check: u_float *. v_float == expected_mul_float
      check: u_float /. v_float == expected_div_float
      check: u_float.asType(Complex[float64]) *. v_float.asType(Complex[float64]) == expected_mul_float.asType(Complex[float64])
      check: u_float.asType(Complex[float64]) /. v_float.asType(Complex[float64]) == expected_div_float.asType(Complex[float64])

    test "Explicit broadcasting":
      block:
        let a = 2.bc([3,2])
        check a == [[2,2],
                    [2,2],
                    [2,2]].toTensor()

      block:
        let a = toSeq(1..2).toTensor().reshape(1,2)
        let b = a.bc([2,2])
        check b == [[1,2],
                    [1,2]].toTensor()

      block:
        let a = toSeq(1..2).toTensor().reshape(2,1)
        let b = a.bc([2,2])
        check b == [[1,1],
                    [2,2]].toTensor()

    test "Implicit tensor-tensor broadcasting - basic operations +., -., *., /., ^.":
      block: # Addition
        let a = [0, 10, 20, 30].toTensor().reshape(4,1)
        let b = [0, 1, 2].toTensor().reshape(1,3)

        check: a +. b == [[0, 1, 2],
                          [10, 11, 12],
                          [20, 21, 22],
                          [30, 31, 32]].toTensor
        check: a.asType(Complex[float64]) +. b.asType(Complex[float64]) == [[0, 1, 2],
                          [10, 11, 12],
                          [20, 21, 22],
                          [30, 31, 32]].toTensor.asType(Complex[float64])


      block: # Substraction
        let a = [0, 10, 20, 30].toTensor().reshape(4,1)
        let b = [0, 1, 2].toTensor().reshape(1,3)

        check: a -. b == [[0, -1, -2],
                          [10, 9, 8],
                          [20, 19, 18],
                          [30, 29, 28]].toTensor
        check: a.asType(Complex[float64]) -. b.asType(Complex[float64]) == [[0, -1, -2],
                          [10, 9, 8],
                          [20, 19, 18],
                          [30, 29, 28]].toTensor.asType(Complex[float64])

      block: # Multiplication
        let a = [0, 10, 20, 30].toTensor().reshape(4,1)
        let b = [0, 1, 2].toTensor().reshape(1,3)

        check: a *. b == [[0, 0, 0],
                          [0, 10, 20],
                          [0, 20, 40],
                          [0, 30, 60]].toTensor
        check: a.asType(Complex[float64]) *. b.asType(Complex[float64]) == [[0, 0, 0],
                          [0, 10, 20],
                          [0, 20, 40],
                          [0, 30, 60]].toTensor.asType(Complex[float64])

      block: # Integer division
        let a = [100, 10, 20, 30].toTensor().reshape(4,1)
        let b = [2, 5, 10].toTensor().reshape(1,3)

        check: a /. b == [[50, 20, 10],
                          [5, 2, 1],
                          [10, 4, 2],
                          [15, 6, 3]].toTensor

      block: # Float division
        let a = [100.0, 10, 20, 30].toTensor().reshape(4,1)
        let b = [2.0, 5, 10].toTensor().reshape(1,3)

        check: a /. b == [[50.0, 20, 10],
                          [5.0, 2, 1],
                          [10.0, 4, 2],
                          [15.0, 6, 3]].toTensor
        check: a.asType(Complex[float64]) /. b.asType(Complex[float64]) == [[50.0, 20, 10],
                          [5.0, 2, 1],
                          [10.0, 4, 2],
                          [15.0, 6, 3]].toTensor.asType(Complex[float64])

      block: # Float division
        var a = [100.0, 10, 20, 30].toTensor().reshape(4,1)
        a /.= 10.0

        check: a == [[10.0],
                    [1.0],
                    [2.0],
                    [3.0]].toTensor

      block: # Float Exponentiation
        let a = [1.0, 10, 20, 30].toTensor().reshape(4,1)

        check: a ^. 2.0 == [[1.0],
                            [100.0],
                            [400.0],
                            [900.0]].toTensor

        check: a ^. -1 == [[1.0],
                            [1.0/10.0],
                            [1.0/20.0],
                            [1.0/30.0]].toTensor

        check: 2.0 ^. a == [[pow(2.0, 1.0)],
                            [pow(2.0, 10.0)],
                            [pow(2.0, 20.0)],
                            [pow(2.0, 30.0)]].toTensor

        check: a.asType(Complex[float64]) ^. complex64(-1.0,0.0) == [[1.0],
                            [1.0/10.0],
                            [1.0/20.0],
                            [1.0/30.0]].toTensor.asType(Complex[float64])

      block: # Modulo operation
        let a = [1, 10, 20, 30].toTensor().reshape(4,1)
        let b = [2, 3, 4].toTensor().reshape(1,3)

        check: a mod b == [[1, 1, 1],
                          [0, 1, 2],
                          [0, 2, 0],
                          [0, 0, 2]].toTensor

    test "Implicit tensor-tensor broadcasting - basic in-place operations +.=, -.=, *.=, /.=":
      block: # Addition
        # Note: We can't broadcast the lhs with in-place operations
        var a = [0, 10, 20, 30].toTensor().reshape(4,1).bc([4,3]).asContiguous
        var a_c = [0, 10, 20, 30].toTensor().reshape(4,1).bc([4,3]).asContiguous.asType(Complex[float64])
        let b = [0, 1, 2].toTensor().reshape(1,3)

        a +.= b
        a_c +.= b.asType(Complex[float64])
        check: a == [[0, 1, 2],
                    [10, 11, 12],
                    [20, 21, 22],
                    [30, 31, 32]].toTensor
        check: a_c == [[0, 1, 2],
                    [10, 11, 12],
                    [20, 21, 22],
                    [30, 31, 32]].toTensor.asType(Complex[float64])

      block: # Substraction
        # Note: We can't broadcast the lhs with in-place operations
        var a = [0, 10, 20, 30].toTensor().reshape(4,1).bc([4,3]).asContiguous
        var a_c = [0, 10, 20, 30].toTensor().reshape(4,1).bc([4,3]).asContiguous.asType(Complex[float64])
        let b = [0, 1, 2].toTensor().reshape(1,3)

        a -.= b
        a_c -.= b.asType(Complex[float64])
        check: a == [[0, -1, -2],
                      [10, 9, 8],
                      [20, 19, 18],
                      [30, 29, 28]].toTensor
        check: a_c == [[0, -1, -2],
                      [10, 9, 8],
                      [20, 19, 18],
                      [30, 29, 28]].toTensor.asType(Complex[float64])

      block: # Multiplication
        # Note: We can't broadcast the lhs with in-place operations
        var a = [0, 10, 20, 30].toTensor().reshape(4,1).bc([4,3]).asContiguous
        var a_c = [0, 10, 20, 30].toTensor().reshape(4,1).bc([4,3]).asContiguous.asType(Complex[float64])
        let b = [0, 1, 2].toTensor().reshape(1,3)

        a *.= b
        a_c *.= b.asType(Complex[float64])
        check: a == [[0, 0, 0],
                    [0, 10, 20],
                    [0, 20, 40],
                    [0, 30, 60]].toTensor
        check: a_c == [[0, 0, 0],
                    [0, 10, 20],
                    [0, 20, 40],
                    [0, 30, 60]].toTensor.asType(Complex[float64])

      block: # Integer division
        # Note: We can't broadcast the lhs with in-place operations
        var a = [100, 10, 20, 30].toTensor().reshape(4,1).bc([4,3]).asContiguous
        let b = [2, 5, 10].toTensor().reshape(1,3)

        a /.= b
        check: a == [[50, 20, 10],
                    [5, 2, 1],
                    [10, 4, 2],
                    [15, 6, 3]].toTensor

      block: # Float division
        # Note: We can't broadcast the lhs with in-place operations
        var a = [100.0, 10, 20, 30].toTensor().reshape(4,1).bc([4,3]).asContiguous
        var a_c = [100.0, 10, 20, 30].toTensor().reshape(4,1).bc([4,3]).asContiguous.asType(Complex[float64])
        let b = [2.0, 5, 10].toTensor().reshape(1,3)

        a /.= b
        a_c /.= b.asType(Complex[float64])
        check: a == [[50.0, 20, 10],
                    [5.0, 2, 1],
                    [10.0, 4, 2],
                    [15.0, 6, 3]].toTensor
        check: a_c == [[50.0, 20, 10],
                    [5.0, 2, 1],
                    [10.0, 4, 2],
                    [15.0, 6, 3]].toTensor.asType(Complex[float64])

    test "Implicit tensor-scalar broadcasting - basic operations +, -, *, /, mod":
      block: # Addition
        var a = [0, 10, 20, 30].toTensor().reshape(4,1)
        var a_c = [0, 10, 20, 30].toTensor().reshape(4,1).asType(Complex[float64])

        a = 100 +. a +. 200
        a_c = complex64(100.0, 0.0) +. a_c +. complex64(200.0, 0.0)
        check: a == [[300],
                    [310],
                    [320],
                    [330]].toTensor
        check: a_c == [[300],
                      [310],
                      [320],
                      [330]].toTensor.asType(Complex[float64])

      block: # Substraction
        var a = [0, 10, 20, 30].toTensor().reshape(4,1)
        var a_c = [0, 10, 20, 30].toTensor().reshape(4,1).asType(Complex[float64])

        a = 100 -. a -. 200
        a_c = complex64(100.0, 0.0) -. a_c -. complex64(200.0, 0.0)
        check: a == [[-100],
                    [-110],
                    [-120],
                    [-130]].toTensor
        check: a_c == [[-100],
                      [-110],
                      [-120],
                      [-130]].toTensor.asType(Complex[float64])

      block: # Multiplication
        var a = [0, 10, 20, 30].toTensor().reshape(4,1)
        var a_c = [0, 10, 20, 30].toTensor().reshape(4,1).asType(Complex[float64])

        a = 10 *. a *. 20
        a_c = complex64(10.0, 0.0) *. a_c *. complex64(20.0, 0.0)
        check: a == [[0],
                    [2000],
                    [4000],
                    [6000]].toTensor
        check: a_c == [[0],
                      [2000],
                      [4000],
                      [6000]].toTensor.asType(Complex[float64])

      block: # Division
        var a = [0, 10, 20, 30].toTensor().reshape(4,1)
        var a_c = [0, 10, 20, 30].toTensor().reshape(4,1).asType(Complex[float64])

        a = (a /. 2) / 5
        a_c = (a_c /. complex[float64](2.0)) / complex[float64](5.0)
        check: a == [[0],
                    [1],
                    [2],
                    [3]].toTensor
        check: a_c == [[0],
                      [1],
                      [2],
                      [3]].toTensor.asType(Complex[float64])

      block: # Modulo operation
        let a = [2, 5, 10].toTensor().reshape(1,3)

        check: a mod 3 == [[2, 2, 1]].toTensor
        check: 3 mod a == [[1, 3, 3]].toTensor

    test "Implicit tensor-scalar broadcasting - basic operations +.=, -.=, .^=":
      block: # Addition
        var a = [0, 10, 20, 30].toTensor().reshape(4,1)
        var a_c = [0, 10, 20, 30].toTensor().reshape(4,1).asType(Complex[float64])

        a +.= 100
        a_c +.= complex64(100.0, 0.0)
        check: a == [[100],
                    [110],
                    [120],
                    [130]].toTensor
        check: a_c == [[100],
                    [110],
                    [120],
                    [130]].toTensor.asType(Complex[float64])

      block: # Substraction
        var a = [0, 10, 20, 30].toTensor().reshape(4,1)
        var a_c = [0, 10, 20, 30].toTensor().reshape(4,1).asType(Complex[float64])

        a -.= 100
        a_c -.= complex64(100.0, 0.0)
        check: a == [[-100],
                      [-90],
                      [-80],
                      [-70]].toTensor
        check: a_c == [[-100],
                      [-90],
                      [-80],
                      [-70]].toTensor.asType(Complex[float64])

      block: # Float Exponentiation
        var a = [1.0, 10, 20, 30].toTensor().reshape(4,1)
        var b = a.clone
        var a_c = a.clone.asType(Complex[float64])
        var b_c = b.clone.asType(Complex[float64])

        a ^.= 2.0
        a_c ^.= complex(2.0)
        check: a == [[1.0],
                      [100.0],
                      [400.0],
                      [900.0]].toTensor
        check: a_c == [[1.0],
                      [100.0],
                      [400.0],
                      [900.0]].toTensor.asType(Complex[float64])

        b ^.= -1
        b_c ^.= complex(-1.0)
        check: b == [[1.0],
                      [1.0/10.0],
                      [1.0/20.0],
                      [1.0/30.0]].toTensor
        check: b_c == [[1.0],
                      [1.0/10.0],
                      [1.0/20.0],
                      [1.0/30.0]].toTensor.asType(Complex[float64])

    test "Implicit tensor-scalar broadcasting - Tensor[Complex64] - scalar operations":
      block:
        let a_c = [10, 20, 30, 40].toTensor().reshape(4,1).asType(Complex[float64])

        check: complex(2.0) +. a_c +. complex(3.0) == 2 +. a_c +. 3
        check: complex(2.0) -. a_c -. complex(3.0) == 2 -. a_c -. 3
        check: complex(2.0) *. a_c *. complex(3.0) == 2 *. a_c *. 3
        check: complex(2.0) /. a_c /. complex(3.0) == 2 /. a_c /. 3
        check: complex(0.5) ^. a_c ^. complex(2.0) == 0.5 ^. a_c ^. 2

      block:
        var a_c = complex([10.0, 20.0, 30.0, 40.0].toTensor,
                          [1.0, 2.0, 3.0, 4.0].toTensor)

        a_c +.= 2
        a_c -.= 3.0
        a_c *.= 4
        a_c /.= 2.0
        a_c ^.= 2
        let expected = complex([320.0, 1428.0, 3328.0, 6020.0].toTensor,
                               [72.0, 304.0, 696.0, 1248.0].toTensor)
        check: a_c.mean_absolute_error(expected) < 1e-12

    test "Implicit broadcasting - Sigmoid 1 ./ (1 +. exp(-x)":
      block:
        proc sigmoid[T: SomeFloat](t: Tensor[T]): Tensor[T]=
          1.T /. (1.T +. exp(0.T -. t))

        proc sigmoid(t: Tensor[Complex32]): Tensor[Complex32]=
          complex32(1) /. (complex32(1) +. exp(complex32(0) -. t))

        let a = newTensor[float32]([2,2])
        check: sigmoid(a) == [[0.5'f32, 0.5],[0.5'f32, 0.5]].toTensor

        let a_c = newTensor[Complex[float32]]([2,2])
        check: sigmoid(a_c) == [[0.5'f32, 0.5],[0.5'f32, 0.5]].toTensor.asType(Complex[float32])

main()
GC_fullCollect()
