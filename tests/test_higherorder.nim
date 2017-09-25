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

import ../src/arraymancer
import unittest, math, future, sequtils

suite "Testing higher-order functions":
  let t = [[0, 1, 2],
          [3, 4, 5],
          [6, 7, 8],
          [9, 10, 11]].toTensor()

  proc customAdd[T: SomeNumber](x, y: T): T = x + y

  test "Map function":

    let t2 = [0, 1, 4, 9, 16, 25, 36, 49, 64, 81, 100, 121].toTensor.reshape([4,3])

    check: t.map(x => x*x) == t2

  test "Apply functions - with in-place and out of place closure":
    var t = toSeq(0..11).toTensor().reshape([4,3])
    let t2 = toSeq(1..12).toTensor().reshape([4,3])

    var tmp1 = t
    tmp1.apply(x => x+1) # out of place
    check: tmp1 == t2

    var tmp2 = t[_,2]

    proc plus_one[T](x: var T) = x += 1
    tmp2.apply(plus_one) # in-place
    check: tmp2 == t2[_,2]

  test "Reduce function":
    check: t.reduce(customAdd) == 66

    proc customConcat(x, y: string): string = x & y

    check: t.map(x => $x).reduce(customConcat) == "01234567891011"

  test "Reduce over an axis":
    proc customMin[T: SomeNumber](x,y: Tensor[T]): Tensor[T] = x - y

    check: t.reduce(customMin, axis = 0) == [-18, -20, -22].toTensor.reshape([1,3])

  test "Fold with different in and result types":
    proc isEven(n: int): bool =
      return n mod 2 == 0

    # Check if all even
    check: t.fold(true, proc(x: bool,y: int): bool = x and y.isEven) == false

    check: (t * 2).fold(true, proc(x: bool,y: int): bool = x and y.isEven) == true

  test "Fold over axis":
    let col_sum_plus_1010 = [[4],
                            [12],
                            [22],
                            [30]].toTensor()

    let initval = [1,0,1,0].toTensor.reshape([4,1])

    check: t.fold(initval, `+`, axis = 1) == col_sum_plus_1010

  test "Fold over two tensors":
    let a = [[1,2],[3,4]].toTensor()
    let b = [[5,6],[7,8]].toTensor()

    let res = a.fold2(0, (r, x, y) => r + x + y, b)
    check res == 36
