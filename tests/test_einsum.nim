# Copyright (c) 2019 the Arraymancer contributors
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).

import ../src/arraymancer
import unittest, sequtils

suite "Einsum":
  test "Transposition of a tensor":
    let a = toSeq(0 .. 5).toTensor.reshape([2, 3]).asType(float)
    let b = einsum(a):
      a[j,i] = a[i,j]
    let res = [[ 0.0,  3.0],
               [ 1.0,  4.0],
               [ 2.0,  5.0]].toTensor
    echo b.shape
    doAssert res == b

  test "Contraction of a whole tensor":
    let a = toSeq(0 .. 5).toTensor.reshape([2, 3]).asType(float)
    let b = einsum(a):
      res = a[i,j]
    let res = 15.0
    echo b
    doAssert res == b

  test "Contraction of a column":
    let a = toSeq(0 .. 5).toTensor.reshape([2, 3]).asType(float)
    let b = einsum(a):
      res[j] = a[i,j]
    let res = [3.0, 5.0, 7.0].toTensor
    echo b
    doAssert res == b

  test "Contraction of a row":
    let a = toSeq(0 .. 5).toTensor.reshape([2, 3]).asType(float)
    let b = einsum(a):
      res[i] = a[i,j]
    let res = [3.0, 12.0].toTensor
    echo b
    doAssert res == b

  test "Matrix-vector multiplication ~ implicit":
    let m = toSeq(0 .. 5).toTensor.reshape([2, 3]).asType(float)
    let v = toSeq(0 .. 2).toTensor.asType(float)
    let b = einsum(m, v):
      m[i,k] * v[k]
    let res = [5.0, 14.0].toTensor
    doAssert res == b

  test "Matrix-vector multiplication ~ explicit":
    let m = toSeq(0 .. 5).toTensor.reshape([2, 3]).asType(float)
    let v = toSeq(0 .. 2).toTensor.asType(float)
    let b = einsum(m, v):
      res[i] = m[i,k] * v[k]
    let res = [5.0, 14.0].toTensor
    doAssert res == b

  test "Matrix-matrix multiplication ~ implicit":
    # TODO: Fix the order of the implicit loops
    let m = toSeq(0 .. 5).toTensor.reshape([2, 3]).asType(float)
    let n = toSeq(0 .. 14).toTensor.reshape([3, 5]).asType(float)
    let b = einsum(m, n):
      m[i,k] * n[k, j]
    let res = [[  25.0,   28.0,   31.0,   34.0,   37.0],
               [  70.0,   82.0,   94.0,  106.0,  118.0]].toTensor
    doAssert res == b.transpose

  test "Matrix-matrix multiplication ~ explicit":
    let m = toSeq(0 .. 5).toTensor.reshape([2, 3]).asType(float)
    let n = toSeq(0 .. 14).toTensor.reshape([3, 5]).asType(float)
    let b = einsum(m, n):
      res[i,j] = m[i,k] * n[k, j]
    let res = [[  25.0,   28.0,   31.0,   34.0,   37.0],
               [  70.0,   82.0,   94.0,  106.0,  118.0]].toTensor
    echo res
    echo b
    doAssert res == b
