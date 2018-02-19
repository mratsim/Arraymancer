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

# Please compile with -d:opencl switch
import ../../src/arraymancer
import unittest

suite "OpenCL BLAS operations (Basic Linear Algebra Subprograms)":
  test "Matrix and vector addition":
    let u = @[1'f32, 3, -5].toTensor.opencl
    let v = @[1'f32, 1, 1].toTensor.opencl

    check: (u + v).cpu == @[2'f32, 4, -4].toTensor()

    let a = @[7.0, 4.0, 3.0, 1.0, 8.0, 6.0, 8.0, 1.0, 6.0, 2.0].toTensor.reshape([5,2]).opencl
    let b = @[6.0, 6.0, 2.0, 0.0, 4.0, 3.0, 2.0, 0.0, 0.0, 3.0].toTensor.reshape([5,2]).opencl

    let apb = @[13.0, 10.0, 5.0, 1.0, 12.0, 9.0, 10.0, 1.0, 6.0, 5.0].toTensor.reshape([5,2])

    check: (a + b).cpu == apb

    # Check size mismatch
    expect(ValueError):
      discard a + b.cpu[0..1, 0..1].opencl

  test "Matrix and vector substraction":
    let u = @[1'f32, 3, -5].toTensor.opencl
    let v = @[1'f32, 1, 1].toTensor.opencl

    check: (u - v).cpu == @[0'f32, 2, -6].toTensor()

    let a = @[7.0, 4.0, 3.0, 1.0, 8.0, 6.0, 8.0, 1.0, 6.0, 2.0].toTensor.reshape([5,2]).opencl
    let b = @[6.0, 6.0, 2.0, 0.0, 4.0, 3.0, 2.0, 0.0, 0.0, 3.0].toTensor.reshape([5,2]).opencl

    let amb = @[1.0, -2.0, 1.0, 1.0, 4.0, 3.0, 6.0, 1.0, 6.0, -1.0].toTensor.reshape([5,2])

    check: (a - b).cpu == amb

    # Check size mismatch
    expect(ValueError):
      discard a + b.cpu[0..1, 0..1].opencl

  test "Scalar/dot product":
    let u = @[float32 1, 3, -5].toTensor().opencl()
    let v = @[float32 4, -2, -1].toTensor().opencl()

    check: dot(u,v) == 3.0'f32


  test "Matrix and Vector in-place addition":
    var u = @[1'f64, 3, -5].toTensor().opencl()
    let v = @[4'f64, -2, -1].toTensor().opencl()

    u += v

    check: u.cpu() == @[5'f64, 1, -6].toTensor()


    # Check require var input
    let w = @[1'f64, 3, -5].toTensor().opencl()
    when compiles(w += v):
      check: false


    let vandermonde = [[1,1,1],
                       [2,4,8],
                       [3,9,27]]

    let t = vandermonde.toTensor.astype(float32)

    var z = t.transpose.opencl()
    z += z

    check: z.cpu == [[2,4,6],
                     [2,8,18],
                     [2,16,54]].toTensor.astype(float32)

    let t2 = vandermonde.toTensor.astype(float32).opencl
    z += t2

    check: z.cpu == [[3,5,7],
                     [4,12,26],
                     [5,25,81]].toTensor.astype(float32)

    # Check size mismatch
    expect(ValueError):
      z += t2.cpu[0..1,0..1].opencl

  test "Matrix and Vector in-place substraction":
    var u = @[1'f32, 3, -5].toTensor.opencl
    let v = @[1'f32, 1, 1].toTensor.opencl

    u -= v

    check: u.cpu == @[0'f32, 2, -6].toTensor()

    # Check require var input
    let w = @[1'f64, 3, -5].toTensor().opencl()
    when compiles(w -= v):
      check: false

    var a = @[7.0, 4.0, 3.0, 1.0, 8.0, 6.0, 8.0, 1.0, 6.0, 2.0].toTensor.reshape([5,2]).opencl
    let b = @[6.0, 6.0, 2.0, 0.0, 4.0, 3.0, 2.0, 0.0, 0.0, 3.0].toTensor.reshape([5,2]).opencl

    let amb = @[1.0, -2.0, 1.0, 1.0, 4.0, 3.0, 6.0, 1.0, 6.0, -1.0].toTensor.reshape([5,2])

    a -= b

    check: a.cpu == amb

    # Check size mismatch
    expect(ValueError):
      a += b.cpu[0..1,0..1].opencl

  test "Matrix and vector addition":
    let u = @[1'f32, 3, -5].toTensor.opencl
    let v = @[1'f32, 1, 1].toTensor.opencl

    check: (u + v).cpu == @[2'f32, 4, -4].toTensor()

    let a = @[7.0, 4.0, 3.0, 1.0, 8.0, 6.0, 8.0, 1.0, 6.0, 2.0].toTensor.reshape([5,2]).opencl
    let b = @[6.0, 6.0, 2.0, 0.0, 4.0, 3.0, 2.0, 0.0, 0.0, 3.0].toTensor.reshape([5,2]).opencl

    let apb = @[13.0, 10.0, 5.0, 1.0, 12.0, 9.0, 10.0, 1.0, 6.0, 5.0].toTensor.reshape([5,2])

    check: (a + b).cpu == apb

    # Check size mismatch
    expect(ValueError):
      discard a + b.cpu[0..1, 0..1].opencl

  test "Matrix and vector substraction":
    let u = @[1'f32, 3, -5].toTensor.opencl
    let v = @[1'f32, 1, 1].toTensor.opencl

    check: (u - v).cpu == @[0'f32, 2, -6].toTensor()

    let a = @[7.0, 4.0, 3.0, 1.0, 8.0, 6.0, 8.0, 1.0, 6.0, 2.0].toTensor.reshape([5,2]).opencl
    let b = @[6.0, 6.0, 2.0, 0.0, 4.0, 3.0, 2.0, 0.0, 0.0, 3.0].toTensor.reshape([5,2]).opencl

    let amb = @[1.0, -2.0, 1.0, 1.0, 4.0, 3.0, 6.0, 1.0, 6.0, -1.0].toTensor.reshape([5,2])

    check: (a - b).cpu == amb

    # Check size mismatch
    expect(ValueError):
      discard a + b.cpu[0..1, 0..1].opencl