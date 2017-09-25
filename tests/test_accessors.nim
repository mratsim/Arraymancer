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
import unittest, math


suite "Accessing and setting tensor values":
  test "Accessing and setting a single value":
    var a = zeros[int](@[2,3,4])
    a[1,2,2] = 122
    check: a[1,2,2] == 122

    var b = zeros[int](@[3,4])
    b[1,2] = 12
    check: b[1,2] == 12
    b[0,0] = 999
    check: b[0,0] == 999
    b[2,3] = 111
    check: b[2,3] == 111


  test "Out of bounds checking":
    var a = newTensor[int](@[2,3,4])
    expect(IndexError):
      a[2,0,0] = 200
    var b = newTensor[int](@[3,4])
    expect(IndexError):
      b[3,4] = 999
    expect(IndexError):
      discard b[-1,0]
    expect(IndexError):
      discard b[0,-2]

  test "Iterators":
    const
      a = @[1, 2, 3, 4, 5]
      b = @[1, 2, 3]
    var
      vd: seq[seq[int]]
      row: seq[int]
    vd = newSeq[seq[int]]()
    for i, aa in a:
      row = newSeq[int]()
      vd.add(row)
      for j, bb in b:
        vd[i].add(aa^bb)

    let nda_vd = vd.toTensor()

    let expected_seq = @[1,1,1,2,4,8,3,9,27,4,16,64,5,25,125]

    var seq_val: seq[int] = @[]
    for i in nda_vd:
      seq_val.add(i)

    check: seq_val == expected_seq

    var seq_validx: seq[tuple[idx: seq[int], val: int]] = @[]
    for i,j in nda_vd:
      seq_validx.add((i,j))

    check: seq_validx[0] == (@[0,0], 1)
    check: seq_validx[10] == (@[3,1], 16)

    let t_nda = transpose(nda_vd)

    var seq_transpose: seq[tuple[idx: seq[int], val: int]] = @[]
    for i,j in t_nda:
      seq_transpose.add((i,j))

    check: seq_transpose[0] == (@[0,0], 1)
    check: seq_transpose[8] == (@[1,3], 16)

  test "indexing + in-place operator":
    var a = newTensor[int]([3,3])

    a[1,1] += 10

    a[1,1] *= 20

    check: a == [[0,0,0],[0,200,0],[0,0,0]].toTensor