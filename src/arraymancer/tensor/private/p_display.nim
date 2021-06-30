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

import  ../../private/functional, ../higher_order_applymap,
        ../shapeshifting, ../data_structure, p_accessors,
        ../accessors,
        sugar, sequtils, strutils

proc bounds_display(t: Tensor,
                    idx_data: tuple[val: string, idx: int],
                    alignBy, alignSpacing: int,
          ): string {.noSideEffect.}=
  ## Internal routine, compare an index with the strides of a Tensor
  ## to check beginning and end of lines
  ## Add the delimiter "|" and line breaks at beginning and end of lines
  ##
  ## `alignBy` is the total fill for each "column" in a 2D print. `alignSpacing`
  ## is the spacing that is ``added`` to the largest element in the original tensor.
  ## Need it to remove that in the first column of each row.
  let (val,idx) = idx_data
  let s = t.shape.reversed

  if val == "|":
    return " | "

  for i,j in s[0 .. s.len-2]: # We don't take the last element (the row in C convention)
    if idx mod j == 0:
      return "" & align(val, alignBy) & "|\n"
    if idx mod j == 1:
      # for the first element  we want to align by only the size of the "largest value" in
      # the tensor, ``not`` the additional spacing we add to space ``between`` numbers.
      # `alignSpacing` is that additional space.
      return "|" & alignLeft(val, alignBy - alignSpacing)

  return "" & align(val, alignBy)

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
# let a = toSeq(1..24).toTensor.reshape(2,3,4)
# echo a
# let b = toSeq(1..72).toTensor.reshape(2,3,3,4)
# echo b

proc disp2d*(t: Tensor, alignBy = 6, alignSpacing = 3): string {.noSideEffect.} =
  ## Display a 2D-tensor

  # Add a position index to each value in the Tensor.
  var indexed_data: seq[(string,int)] = @[]
  for i, value in t.enumerate:
    indexed_data.add(($value,i+1))  # TODO Note: the $conversion is unstable if the whole test suite is done.
                                    # it fails at the test_load_openmp.
                                    # if done alone there is no issue

  # Create a closure to apply the boundaries transformation for the specific input
  proc curry_bounds(tup: (string,int)): string {.noSideEffect.} =
    t.bounds_display(tup, alignBy = alignBy, alignSpacing = alignSpacing)

  return indexed_data.concatMap(curry_bounds)

proc disp3d*(t: Tensor): string =
  ## Display a 3D-tensor

  let sep: seq[string] = @["|"]
  let empty: seq[string] = @[""]

  var buffer = empty.repeat(t.shape[1]).toTensor()

  {.push hint[Pattern]: off.} # Remove Pattern hint for toTensor.reshape
  for t0 in t.axis(0):
    buffer = buffer.concat(
              sep.repeat(t0.shape[1]).toTensor().reshape(t.shape[1],1),
              t0.map((x:type(t0[0]))->string => $x).reshape(t.shape[1], t.shape[2]),
              axis = 1
              )

  return buffer.disp2d

proc disp4d*(t: Tensor): string =
  ## Display a 4D-tensor

  let sep: seq[string] = @["|"]
  let sepv: seq[string] = @["-"]
  let empty: seq[string] = @[""]

  # First create seq of tensor to concat horizontally
  var hbuffer = newSeqWith(t.shape[0], empty.repeat(t.shape[2]).toTensor())

  {.push hint[Pattern]: off.} # Remove Pattern hint for toTensor.reshape
  var i = 0
  for s0 in t.axis(0):
    let s0r = s0.reshape(t.shape[1],t.shape[2],t.shape[3])
    for s1 in s0r.axis(0):
      hbuffer[i] = hbuffer[i].concat(
                sep.repeat(t.shape[2]).toTensor().reshape(t.shape[2],1),
                s1.reshape(t.shape[2], t.shape[3]).map((x:type(s1[0]))->string => $x),
                axis = 1
                )
    inc i


  # Then concat vertically
  var vbuffer = empty.repeat(hbuffer[0].shape[1]).toTensor.reshape(1, hbuffer[0].shape[1])

  for h in hbuffer:
    vbuffer = vbuffer.concat(
              sepv.repeat(hbuffer[0].shape[1]).toTensor().reshape(1, hbuffer[0].shape[1]),
              h.map((x:type(h[0]))->string => $x).reshape(hbuffer[0].shape[0], hbuffer[0].shape[1]),
              axis = 0
              )
  return vbuffer.disp2d

proc zipStrings(s1, s2: string, sep = "", allowEmpty = false): string =
  ## zips two strings line by line to a combined string
  let s1S = s1.splitLines
  let s2S = s2.splitLines
  if s1S.len == 1: return s2
  elif s2S.len == 1: return s1
  for (x, y) in zip(s1S, s2S):
    if not allowEmpty and (x.len == 0 and y.len == 0):
      continue
    result.add $x & $sep & $y & "\n"

proc genSep(rank: int, lineLen = 0, xaxis = false): string =
  ## generate horizontal / vertical separator lines based on the axis and tensor rank
  var sepLine = ""
  let drawInEven = rank mod 2 == 0
  for i in 2 ..< rank:
    if drawInEven and i mod 2 == 0:
      if not xaxis:
        sepLine.add " "
      else:
        sepLine.add repeat("-", lineLen)
    elif drawInEven and i mod 2 != 0:
      if not xaxis:
        sepLine.add " | "
      else:
        sepLine.add repeat(" ", lineLen)
    elif not drawInEven and i mod 2 == 0:
      if not xaxis:
        sepLine.add " "
      else:
        sepLine.add repeat("-", lineLen)
    else:
      if not xaxis:
        sepLine.add " | "
      else:
        sepLine.add repeat(" ", lineLen)
    if i < rank and xaxis:
      sepLine.add "\n"
  result = sepLine

proc genLeftIdx(axIdx: string, s: string): string =
  ## Take the input, center index in the middle and then split them by whitespace.
  ## Can use this to have row centered entry
  let tmp = center(axIdx & " ", s.splitLines.len).split()
  for i in 0 ..< tmp.high:
    let l = tmp[i]
    if l.len > 0:
      result.add l & " "
    else:
      result.add repeat(" ", ($axIdx).len) & " "
    if i < tmp.high - 1:
      result.add "\n"

proc prettyImpl*[T](t: Tensor[T], inputRank = 0, alignBy = 0, alignSpacing = 4): string =
  ## Pretty printing implementation that aligns N dimensional tensors as a
  ## table. Odd dimensions are stacked horizontally and even dimensions
  ## vertically.
  var alignBy = alignBy
  var inputRank = inputRank
  if inputRank == 0:
    inputRank = t.rank
    let largestElement = t.map_inline(($x).len).max
    alignBy = max(6, largestElement + alignSpacing)
  # for tensors of rank larger 2, walk axis 0 and stack vertically (even dim)
  # or stack horizontally (odd dim)
  if t.rank > 2:
    var axIdx = 0
    var res = ""
    let oddRank = t.rank mod 2 != 0
    for ax in axis(t, 0):
      if oddRank:
        # 1. get next "column"
        var toZip = prettyImpl(ax.squeeze, inputRank, alignBy = alignBy)
        # 2. center current "column" index to width of `toZip`, put on top
        toZip = center($axIdx, toZip.splitLines[0].len) & "\n" & toZip
        # 3. generate separator of "columns" and zip together
        let sep = t.rank.genSep()
        res = res.zipStrings(toZip, sep = sep, allowEmpty = false)
      else:
        # 1. get next "row"
        var toStack = prettyImpl(ax.squeeze, inputRank, alignBy = alignBy)
        # 2. center current "row" index to height of `toStack`
        let leftIdx = genLeftIdx($axIdx, toStack)
        # 3. zip index and "row"
        toStack = zipStrings(leftIdx, toStack, allowEmpty = true)
        # 4. stack on top of current result
        res.add toStack
      inc axIdx
    # finally add a horizontal separator if we are not at "top" level
    if t.rank mod 2 != 0 and t.rank != inputRank:
      let sepLine = t.rank.genSep(res.splitLines[0].len, true)
      res.add sepLine
    result.add res
  else:
    result = t.disp2d(alignBy = alignBy,
                      alignSpacing = alignSpacing).strip
