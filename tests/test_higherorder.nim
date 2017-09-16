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
import unittest, math, future

suite "Testing higher-order functions":
  let t = [[0, 1, 2],
          [3, 4, 5],
          [6, 7, 8],
          [9, 10, 11]].toTensor()

  test "Map function":

    let t2 = [0, 1, 4, 9, 16, 25, 36, 49, 64, 81, 100, 121].toTensor.reshape([4,3])

    check: t.map(x => x*x) == t2

  test "Reduce function":
    proc customAdd[T: SomeNumber](x, y: T): T = x + y

    check: t.reduce(customAdd) == 66

    proc customConcat(x, y: string): string = x & y

    check: t.map(x => $x).reduce(customConcat) == "01234567891011"

  test "Reduce over an axis":
    proc customMin[T: SomeNumber](x,y: Tensor[T]): Tensor[T] = x - y

    check: t.reduce(customMin, axis = 0) == [-18, -20, -22].toTensor.reshape([1,3])