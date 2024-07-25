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
        ../shapeshifting, ../data_structure,
        ../accessors
import std / [sequtils, strutils, strformat, typetraits]

# We use different specifier string "tokens" to set the tensor display mode:
# - || or <>||: forces "table mode" (without or with header)
# - [:] or <>[:] (or the equivalent <:>): multi line array mode (w/o or w/ header)
# - [] or <>[] (or the equivalent <>): single line array mode (w/o or w/ header)
# If none of those tokens is found use table mode
#
func isTensorHeaderEnabled(specifier: string): bool =
  ## Return true when a header must be shown based on the format specifier
  # Show a header when "<>" (i.e. the explicit header marker) or "<:>" (i.e. the
  # combined <>[:] shortcut) is found
  # Otherwise do not show a header if "||" or "[]" or "[:]" is found
  # Show a header in all other cases
  # if "<>" in specifier or "<:>" in specifier:
  if "<" in specifier and ">" in specifier:
    # Note that this also covers the "<>[]" and "<>||" cases
    return true
  elif specifier.count('|') == 2 or ("[" in specifier and "]" in specifier):
    return false
  return true

type tensorDispMode = enum table, multi_line_array, single_line_array

func getTensorDispMode(specifier: string): tensorDispMode =
  ## Get the display mode based on the special "tensor specifier tokens"
  let brackets = "[" in specifier and "]" in specifier
  let colon_brackets = specifier.count(':') == 2
  let logic_brackets = "<" in specifier and ">" in specifier
  let flat_brackets = specifier.count('|') == 2
  if flat_brackets:
    # Note that "<>||" also implies "table" mode!
    table
  elif colon_brackets or
      (":" in specifier and (brackets or logic_brackets)):
    # <:> is a shortcut for <>[:]
    multi_line_array
  elif brackets or logic_brackets:
    # <> is a shortcut for <>[] _when_ not combined with ||, ::, [:] or [::]
    single_line_array
  else:
    # In all other cases, default to "table" mode!
    table

func removeTensorFormatSpecifiers(specifier: string): string =
  ## Remove the special "tensor specifier tokens"
  ##
  ## This must be called _after_ getting the display mode (using
  ## `getTensorDispMode`) and the "show header flat" (using
  ## `isTensorHeaderEnabled`), to avoid an error when the specifier
  ## is finally passed to the tensor element `formatValue` functions
  result = specifier.replace("::")
                    .replace("[:]")
                    .replace("<:>")
                    .replace("<>")
                    .replace("[]")
                    .replace("||")

func bounds_display(t: Tensor,
                    idx_data: tuple[val: string, idx: int],
                    alignBy, alignSpacing: int
          ): string =
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
      if t.rank == 2 and t.shape[1] == 1:
        return "|" & align(val, alignBy - alignSpacing) & "|\n"
      else:
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

func dispElement*[T](value: T, precision = -1, specifier: static string = ""): string =
  ## Display a single element with the selected precision _or_ specifier format
  when specifier.len == 0:
    when T is SomeFloat:
      result = formatBiggestFloat(value, precision = precision)
    else:
      result = $value
  else:
    formatValue(result, value, specifier = specifier)

func disp2d*[T](t: Tensor[T], alignBy = 6, alignSpacing = 3,
                precision = -1, specifier: static string = ""): string =
  ## Display a 2D-tensor (only used for "table", i.e. non array, printing)

  # Add a position index to each value in the Tensor.
  var indexed_data: seq[(string,int)] = @[]
  for i, value in t.enumerate:
    let val = dispElement(value, precision = precision, specifier = specifier)
    indexed_data.add((val, i+1))  # TODO Note: the $conversion is unstable if the whole test suite is done.
                                  # it fails at the test_load_openmp.
                                  # if done alone there is no issue

  # Create a closure to apply the boundaries transformation for the specific input
  func curry_bounds(tup: (string,int)): string =
    t.bounds_display(tup, alignBy = alignBy, alignSpacing = alignSpacing)

  return indexed_data.concatMap(curry_bounds)

func zipStrings(s1, s2: string, sep = "", allowEmpty = false): string =
  ## zips two strings line by line to a combined string
  let s1S = s1.splitLines
  let s2S = s2.splitLines
  if s1S.len == 1: return s2
  elif s2S.len == 1: return s1
  for (x, y) in zip(s1S, s2S):
    if not allowEmpty and (x.len == 0 and y.len == 0):
      continue
    result.add $x & $sep & $y & "\n"

func genSep(rank: int, lineLen = 0, xaxis = false): string =
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

func genLeftIdx(axIdx: string, s: string): string =
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

proc determineLargestElement[T](t: Tensor[T], precision: int, specifier: static string = ""): int =
  ## Determines the length of the "largest" element in the tensor after
  ## string conversion. This is to align our output table nicely.
  # when T is SomeFloat:
  #   result = t.map_inline((x.formatBiggestFloat(precision = precision)).len).max
  # else:
  #   result = t.map_inline(($x).len).max
  result = t.map_inline(x.dispElement(precision = precision, specifier = specifier).len).max

proc dispTensorAsTable*[T](t: Tensor[T],
                           inputRank = 0, alignBy = 0, alignSpacing = 4,
                           precision = -1,
                           specifier: static string = ""): string =
  ## Pretty printing implementation that aligns N dimensional tensors as a
  ## table. Odd dimensions are stacked horizontally and even dimensions
  ## vertically.
  ##
  ## `inputRank` is used to keep track of the original tensor's rank. `alignBy`
  ## is the spacing given each column in a sub-tensor. `alignSpacing` is the
  ## amount of space we want at least between the different columns. It's given
  ## separately for the special case of first columns (as they are left aligned
  ## and all others right aligned).
  ##
  ## `precision` sets the floating point precision.
  const specifier = removeTensorFormatSpecifiers(specifier)
  var alignBy = alignBy
  var inputRank = inputRank
  if inputRank == 0:
    inputRank = t.rank
    let largestElement = t.determineLargestElement(precision, specifier)
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
        var toZip = dispTensorAsTable(ax.squeeze,
                                      inputRank,
                                      alignBy = alignBy,
                                      precision = precision,
                                      specifier = specifier)
        # 2. center current "column" index to width of `toZip`, put on top
        toZip = center($axIdx, toZip.splitLines[0].len) & "\n" & toZip
        # 3. generate separator of "columns" and zip together
        let sep = t.rank.genSep()
        res = res.zipStrings(toZip, sep = sep, allowEmpty = false)
      else:
        # 1. get next "row"
        var toStack = dispTensorAsTable(ax.squeeze,
                                        inputRank,
                                        alignBy = alignBy,
                                        precision = precision,
                                        specifier = specifier)
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
                      alignSpacing = alignSpacing,
                      precision = precision,
                      specifier = specifier).strip

proc disp1dAsArray[T](t: Tensor[T],
                    sep = ", ",
                    precision = -1, specifier: static string = ""): string =
  ## Display a 1D-tensor (only used for "array-style", i.e. non-table, printing)
  if t.len == 0:
    return "[]"
  result = "["
  for value in t:
    result &= dispElement(value, precision = precision, specifier = specifier)
    result &= sep
  # Remove the separator from the last element
  result = result[0..^(1+sep.len)] & "]"
  when T is Complex and "j" in specifier:
    result = result.replace("(").replace(")")

proc compactTensorDescription[T](t: Tensor[T]): string =
  ## Describe the tensor in terms of its shape and type (in a "compact" way)
  ## Only used for array-style printing
  # Most if not all tensors element types are part of the system or complex
  # modules so we can remove them from the type without must information loss
  let compactType = t.type.name().replace("system.", "").replace("complex.", "")
  let compactShape = ($t.shape)[1 ..< ^1].replace(", ", ",")
  result = compactType & "<" & compactShape & ">"

proc squeezeTopDimension[T](t: Tensor[T]): Tensor[T] =
  ## Remove the top most dimension if its size is 1
  if t.shape.len == 0 or t.shape[0] > 1:
    return t
  var new_shape = t.shape
  new_shape.delete(0)
  result = t.reshape(new_shape)

proc dispTensorAsSingleLineArrayImp[T](t: Tensor[T],
                    precision = -1,
                    specifier: static string = "",
                    indentSpacing = 0,
                    sep = ", ", rowSep = ""
                    ): string =
  ## Implementation of the "array-style" tensor printing
  result = "["
  if t.rank <= 1:
    result = disp1dAsArray(t, sep = sep, precision = precision, specifier = specifier)
  else:
    var n = 0
    for ax in axis(t, 0):
      var axRepr = dispTensorAsSingleLineArrayImp(ax.squeezeTopDimension(),
        precision = precision,
        specifier = specifier,
        indentSpacing = indentSpacing,
        sep = sep, rowSep = rowSep)
      result &= axRepr
      n += 1
      if n < t.shape[0]:
        result &= sep & rowSep
    result &= "]"

proc dispTensorAsSingleLineArray*[T](t: Tensor[T],
                    precision = -1,
                    indentSpacing = 0,
                    specifier: static string = "",
                    sep = ", ", rowSep = "",
                    showHeader = true
                    ): string =
  ## Display a tensor as a single line "array"
  # Remove the non-standard specifier flags (n, a and s)
  const specifier = removeTensorFormatSpecifiers(specifier)
  if showHeader:
    result = t.compactTensorDescription & ":" & rowSep
  result &= dispTensorAsSingleLineArrayImp(t, precision, specifier = specifier, rowSep = rowSep)
  if t.storage.isNil:
    # Return a useful message for uninit'd tensors instead of crashing
    # Note that this should only happen when displaying tensors created
    # by just declaring their type (e.g. `var t: Tensor[int]`), given that
    # even tensors created by calling `newTensorUninit` have their storage
    # initialized (with garbage)
    result &= " (uninitialized)"

proc indentTensorReprRows(s: string, indent: int): string =
  ## Indent the lines of a multi-line "array-style" tensor representation
  ## so that the right-most opening braces align vertically
  if indent <= 0:
    return s
  for line in s.splitLines():
    var numBrackets = 0
    for c in line:
      if c != '[':
        break
      numBrackets += 1
    result &= line.indent(indent - numBrackets) & "\n"

proc dispTensorAsArray*[T](t: Tensor[T],
                    precision = -1,
                    specifier: static string = "",
                    showHeader = true): string =
  ## Display a tensor as a multi-line "array"
  result = t.dispTensorAsSingleLineArray(
    precision = precision, specifier = specifier, rowSep="\n", showHeader = false)
  result = indentTensorReprRows(result, t.rank).strip(leading=false)
  if showHeader:
    result = t.compactTensorDescription() & ":\n" & result

proc prettyImpl*[T](
    t: Tensor[T], precision: int, specifier: static string): string =
  ## Non public implementation of the pretty function
  ## Three modes are supported: table, multi-line array and single-line array
  let showHeader = isTensorHeaderEnabled(specifier)
  let dispMode = getTensorDispMode(specifier)
  const specifier = removeTensorFormatSpecifiers(specifier)
  if dispMode == single_line_array:
    return t.dispTensorAsSingleLineArray(
      precision = precision, specifier = specifier, showHeader = showHeader)
  elif dispMode == multi_line_array:
    return t.dispTensorAsArray(
      precision = precision, specifier = specifier, showHeader = showHeader)
  # Represent the tensor as a "pretty" table
  var desc = t.type.name & " of shape \"" & $t.shape & "\" on backend \"" & "Cpu" & "\""
  if t.storage.isNil: # return useful message for uninit'd tensors instead of crashing
    return "Uninitialized " & $desc
  if showHeader:
    desc &= "\n"
  else:
    desc = ""
  if t.size() == 0:
    return desc & "    [] (empty)"
  elif t.rank == 1: # for rank 1 we want an indentation, because we have no `|`
    return desc & "    " & t.dispTensorAsTable(
      precision = precision, specifier = specifier)
  else:
    return desc & t.dispTensorAsTable(
      precision = precision, specifier = specifier)
