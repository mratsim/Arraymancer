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

proc bounds_display(t: Tensor,
          idx_data: tuple[val: string, idx: int]
          ): string {.noSideEffect.}=
  ## Internal routine, compare an index with the strides of a Tensor
  ## to check beginning and end of lines
  ## Add the delimiter "|" and line breaks at beginning and end of lines
  ## TODO: improve 3+D-tensors display
  let (val,idx) = idx_data
  let s = t.shape.reversed

  if val == "|":
    return " | "

  for i,j in s[0 .. ^2]: # We don't take the last element (the row in C convention)
    if idx mod j == 0:
      return "\t" & $val & "|\n"
    if idx mod j == 1:
      return "|" & $val
  return "\t" & $val

# TODO: Create a generic n-dimensional display function using nested tables.
# Example code in hTensor: https://github.com/albertoruiz/hTensor/blob/b36c3748b211c7f41c9af9d486c6ef320e2b7585/lib/Numeric/LinearAlgebra/Array/Display.hs#L92

# Last dim always in column (except vector)
# If rank is odd, first dim is along columns
# if rank is even, first dim is along row

# Expected for 2x3x3
#                      0                                       1
# ---------------------------------------
# 0,0 # 0,0,0    0,0,1    0,0,2    0,0,3 | 1,0 # 1,0,0    1,0,1    1,0,2    1,0,3
# 0,1 # 0,1,0    0,1,1    0,1,2    0,1,3 | 1,1 # 1,1,0    1,1,1    1,1,2    1,1,3
# 0,2 # 0,2,0    0,2,1    0,2,2    0,2,3 | 1,2 # 1,2,0    1,2,1    1,2,2    1,2,3
# ---------------------------------------

# Expected for 2x3x3x4
# 1   2   3   4| 13 14 15 16 | 25 26 27 28
# 5   6   7   8| 17 18 19 20 | 29 30 31 32
# 9  10  11  12| 21 22 23 24 | 33 34 35 36
# ----------------------------------------------
# 37 38  39  40| 49 59 51 52 | 61 62 63 64
# 41 42  43  44| 53 54 55 56 | 65 66 67 68
# 45 46  47  48| 57 58 59 60 | 69 70 71 72

# Test with
# let a = toSeq(1..24).toTensor(Cpu).reshape(2,3,4)
# echo a
# let b = toSeq(1..72).toTensor(Cpu).reshape(2,3,3,4)
# echo b

proc disp2d(t: Tensor): string {.noSideEffect.} =
  ## Display a 2D-tensor

  # Add a position index to each value in the Tensor.
  var indexed_data: seq[(string,int)] = @[]
  for i, value in t.enumerate:
    indexed_data.add(($value,i+1))

  # Create a closure to apply the boundaries transformation for the specific input
  proc curry_bounds(tup: (string,int)): string {.noSideEffect.}= t.bounds_display(tup)

  return indexed_data.concatMap(curry_bounds)

proc disp3d(t: Tensor): string =
  ## Display a 3D-tensor

  let sep: seq[string] = @["|"]
  let empty: seq[string] = @[]

  var buffer = empty.repeat(t.shape[1]).toTensor()

  for t0 in t.axis(0):
    buffer = buffer.concat(
              sep.repeat(t0.shape[1]).toTensor().reshape(t.shape[1],1),
              t0.map(x => $x).reshape(t.shape[1], t.shape[2]),
              axis = 1
              )

  return buffer.disp2d

proc disp4d(t: Tensor): string =
  ## Display a 4D-tensor

  let sep: seq[string] = @["|"]
  let sepv: seq[string] = @["-"]
  let empty: seq[string] = @[]

  # First create seq of tensor to concat horizontally
  var hbuffer = newSeqWith(t.shape[0], empty.repeat(t.shape[2]).toTensor())

  var i = 0
  for s0 in t.axis(0):
    let s0r = s0.reshape(t.shape[1],t.shape[2],t.shape[3])
    for s1 in s0r.axis(0):
      hbuffer[i] = hbuffer[i].concat(
                sep.repeat(t.shape[2]).toTensor().reshape(t.shape[2],1),
                s1.reshape(t.shape[2], t.shape[3]).map(x => $x),
                axis = 1
                )
    inc i

  # Then concat vertically
  var vbuffer = empty.repeat(hbuffer[0].shape[1]).toTensor().reshape(0, hbuffer[0].shape[1])

  for h in hbuffer:
    vbuffer = vbuffer.concat(
              sepv.repeat(hbuffer[0].shape[1]).toTensor().reshape(1, hbuffer[0].shape[1]),
              h.map(x => $x).reshape(hbuffer[0].shape[0], hbuffer[0].shape[1]),
              axis = 0
              )
  return vbuffer.disp2d

proc `$`*[T](t: Tensor[T]): string =
  ## Pretty-print a tensor (when using ``echo`` for example)
  let desc = "Tensor of shape " & t.shape.join("x") & " of type \"" & T.name & "\" on backend \"" & "Cpu" & "\""
  if t.rank <= 2:
    return desc & "\n" & t.disp2d
  elif t.rank == 3:
    return desc & "\n" & t.disp3d
  elif t.rank == 4:
    return desc & "\n" & t.disp4d
  else:
    return desc & "\n" & " -- NotImplemented: Display not implemented for tensors of rank > 4"