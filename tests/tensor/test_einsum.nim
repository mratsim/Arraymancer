# Copyright (c) 2019 the Arraymancer contributors
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).

import ../../src/arraymancer
import unittest, sequtils

# The tests are adapted from here:
# https://rockt.github.io/2018/04/30/einsum

suite "Einsum":
  test "Transposition of a tensor":
    let a = toSeq(0 .. 5).toTensor.reshape([2, 3]).asType(float)
    let b = einsum(a):
      b[j,i] = a[i,j]
    let res = [[ 0.0,  3.0],
               [ 1.0,  4.0],
               [ 2.0,  5.0]].toTensor
    doAssert res == b

  test "Contraction of a whole tensor":
    let a = toSeq(0 .. 5).toTensor.reshape([2, 3]).asType(float)
    let b = einsum(a):
      b = a[i,j]
    let res = 15.0
    doAssert res == b

  test "Contraction of a column":
    let a = toSeq(0 .. 5).toTensor.reshape([2, 3]).asType(float)
    let b = einsum(a):
      b[j] = a[i,j]
    let res = [3.0, 5.0, 7.0].toTensor
    doAssert res == b

  test "Contraction of a row":
    let a = toSeq(0 .. 5).toTensor.reshape([2, 3]).asType(float)
    let b = einsum(a):
      b[i] = a[i,j]
    let res = [3.0, 12.0].toTensor
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
      b[i] = m[i,k] * v[k]
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
    doAssert res == b

  test "Matrix-matrix multiplication ~ explicit":
    let m = toSeq(0 .. 5).toTensor.reshape([2, 3]).asType(float)
    let n = toSeq(0 .. 14).toTensor.reshape([3, 5]).asType(float)
    let b = einsum(m, n):
      b[i,j] = m[i,k] * n[k, j]
    let res = [[  25.0,   28.0,   31.0,   34.0,   37.0],
               [  70.0,   82.0,   94.0,  106.0,  118.0]].toTensor
    doAssert res == b

  test "Dot product ~ implicit":
    let v = toSeq(0 .. 2).toTensor.asType(float)
    let w = toSeq(3 .. 5).toTensor.asType(float)
    let b = einsum(v, w):
      v[i] * w[i]
    let res = 14.0
    doAssert res == b

  test "Dot product ~ explicit":
    let v = toSeq(0 .. 2).toTensor.asType(float)
    let w = toSeq(3 .. 5).toTensor.asType(float)
    let b = einsum(v, w):
      b = v[i] * w[i]
    let res = 14.0
    doAssert res == b

  test "Matrix dot product ~ implicit":
    let m = toSeq(0 .. 5).toTensor.reshape([2, 3]).asType(float)
    let n = toSeq(6 .. 11).toTensor.reshape([2, 3]).asType(float)
    let b = einsum(m, n):
      m[i, j] * n[i, j]
    let res = 145.0
    doAssert res == b

  test "Matrix dot product ~ explcit":
    let m = toSeq(0 .. 5).toTensor.reshape([2, 3]).asType(float)
    let n = toSeq(6 .. 11).toTensor.reshape([2, 3]).asType(float)
    let b = einsum(m, n):
      b = m[i, j] * n[i, j]
    let res = 145.0
    doAssert res == b

  test "Hadamard product":
    let m = toSeq(0 .. 5).toTensor.reshape([2, 3]).asType(float)
    let n = toSeq(6 .. 11).toTensor.reshape([2, 3]).asType(float)
    let b = einsum(m, n):
      b[i, j] = m[i, j] * n[i, j]
    let res = [[  0.0,   7.0,  16.0],
               [ 27.0,  40.0,  55.0]].toTensor
    doAssert res == b

  test "Outer product":
    let v = toSeq(0 .. 2).toTensor.asType(float)
    let w = toSeq(3 .. 6).toTensor.asType(float)
    let b = einsum(v, w):
      b[i, j] = v[i] * w[j]
    let res = [[  0.0,   0.0,   0.0,   0.0],
               [  3.0,   4.0,   5.0,   6.0],
               [  6.0,   8.0,  10.0,  12.0]].toTensor
    doAssert res == b

  test "'Batch' Matrix multiplication (tensor contraction)":
    let m = toSeq(0 .. 29).toTensor.reshape([3, 2, 5]).asType(float)
    let n = toSeq(30 .. 74).toTensor.reshape([3, 5, 3]).asType(float)
    let b = einsum(m, n):
      b[i,j,l] = m[i,j,k] * n[i,k,l]
    let res = [[[ 390.0,  400.0,  410.0],
                [1290.0, 1325.0, 1360.0]],
               [[3090.0, 3150.0, 3210.0],
                [4365.0, 4450.0, 4535.0]],
               [[7290.0, 7400.0, 7510.0],
                [8940.0, 9075.0, 9210.0]]].toTensor
    doAssert res == b

  test "Larger tensor contraction ~ implicit & explicit":
    let m = toSeq(0 .. 209).toTensor.reshape([2, 3, 5, 7]).asType(float)
    let n = toSeq(0 .. 179).toTensor.reshape([2, 3, 3, 2, 5]).asType(float)
    let bImpl = einsum(m, n):
      m[p,q,r,s] * n[t,u,q,v,r]
    let resShape = @[2, 7, 2, 3, 2]
    doAssert resShape == toSeq(bImpl.shape)
    let bExpl = einsum(m, n):
      bExpl[p,s,t,u,v] = m[p,q,r,s] * n[t,u,q,v,r]
    doAssert resShape == toSeq(bExpl.shape)
    doAssert bImpl == bExpl

  test "Bilinear transformation / working with 3 tensors":
    let m = toSeq(0 .. 5).toTensor.reshape([2, 3]).asType(float)
    let n = toSeq(0 .. 5 * 3 * 7 - 1).toTensor.reshape([5, 3, 7]).asType(float)
    let p = toSeq(0 .. 13).toTensor.reshape([2, 7]).asType(float)
    let b = einsum(m, n, p):
      b[i,j] = m[i,k] * n[j,k,l] * p[i,l]
    let res = [[ 1008.0,  2331.0,  3654.0,  4977.0,  6300.0],
               [ 9716.0, 27356.0, 44996.0, 62636.0, 80276.0]].toTensor
    doAssert res == b

  # and now for some tests `pytorch.einsum` apparently cannot do
  test "Diagonal elements":
    let m = toSeq(0 .. 8).toTensor.reshape([3, 3]).asType(float)
    let b = einsum(m):
      b[i] = m[i, i]
    let res = [0.0, 4.0, 8.0].toTensor
    doAssert res == b
