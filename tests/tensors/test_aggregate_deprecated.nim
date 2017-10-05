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

import ../../src/arraymancer
import unittest, math

suite "Testing aggregation functions":
  let t = [[0, 1, 2],
          [3, 4, 5],
          [6, 7, 8],
          [9, 10, 11]].toTensor(Cpu)

  test "Sum all elements":
    check: t.sum == 66

  test "Sum over axis":
    let row_sum = [[18, 22, 26]].toTensor(Cpu)
    let col_sum = [[3],
                   [12],
                   [21],
                   [30]].toTensor(Cpu)
    check: t.sum(axis=0) == row_sum
    check: t.sum(axis=1) == col_sum

    ## TODO: 3D axis sum
  test "Mean of all elements":
    check: t.astype(float).mean == 5.5 # Note: may fail due to float rounding

  test "Mean over axis":
    let row_mean = [[4.5, 5.5, 6.5]].toTensor(Cpu)
    let col_mean = [[1.0],
                   [4.0],
                   [7.0],
                   [10.0]].toTensor(Cpu)
    check: t.astype(float).mean(axis=0) == row_mean
    check: t.astype(float).mean(axis=1) == col_mean

  test "Generic aggregate functions":
    # We can't pass built-ins to procvar
    proc addition[T](a, b: T): T=
      return a+b
    proc addition_inplace[T](a: var T, b: T)=
      a+=b

    check: t.agg(addition, start_val=0) == 66

    var z = 0
    z.agg_inplace(addition_inplace, t)
    check: z == 66

    #### Axis - `+`, `+=` for tensors are not "built-ins"
    let row_sum = [[18, 22, 26]].toTensor(Cpu)
    let col_sum = [[3],
                   [12],
                   [21],
                   [30]].toTensor(Cpu)

    var z1 = zeros([1,3], int, Cpu)
    var z2 = zeros([4,1], int, Cpu)

    # Start with non-inplace proc
    check: t.agg(`+`, axis=0, start_val = z1 ) == row_sum
    check: t.agg(`+`, axis=1, start_val = z2 ) == col_sum

    # Inplace proc
    # z1.agg_inplace(`+=`, t, axis=0)
    # z2.agg_inplace(`+=`, t, axis=1)

    # check: z1 == row_sum
    # check: z2 == col_sum