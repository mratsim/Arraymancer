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


import  os, parsecsv, streams, strutils, sequtils, algorithm,
        ../tensor
from memfiles as mf import nil

proc countLinesAndCols(file: string, sep: char, quote: char,
                       skipHeader: bool): tuple[rows: int, cols: int] =
  ## Counts the number of lines and columns in the given `file`.
  ##
  ## This uses the `memfiles` interface for performance reasons to avoid
  ## unnecessary overhead purely for counting lines. Ideally, the actual
  ## CSV parsing would also use the same interface.
  var memf = mf.open(file)
  defer: mf.close(memf)
  var countedCols = false
  var nCols = 1 # at least 1 column
  var nRows = 0
  var cstr: cstring
  var quoted = false
  for slice in mf.memSlices(memf):
    cstr = cast[cstring](slice.data)
    if slice.size > 0 and unlikely(not countedCols): # count number of columns
      for idx in 0 ..< slice.size:                   # need to be careful to only access to `size`
        if cstr[idx] == sep:                         # a separator means another column
          inc nCols
      inc nRows
      countedCols = true
    elif slice.size > 0:                             # only count non empty lines from here
      for idx in 0 ..< slice.size:
        if cstr[idx] == quote:
          quoted = not quoted
      if not quoted:
        inc nRows
  if skipHeader:
    dec nRows
  result = (rows: nRows, cols: nCols)

proc read_csv*[T: SomeNumber|bool|string](
       csvPath: string,
       skipHeader = false,
       separator = ',',
       quote = '\"'
       ): Tensor[T] {.noinit.} =
  ## Load a csv into a Tensor. All values must be of the same type.
  ##
  ## If there is a header row, it can be skipped.
  ##
  ## The reading of CSV files currently ``does not`` handle parsing a tensor
  ## created with `toCsv`. This is because the dimensional information becomes
  ## part of the CSV output and the parser has no option to reconstruct the
  ## correct tensor shape.
  ## Instead of a NxMx...xZ tensor we always construct a NxM tensor, where N-1
  ## is the rank of the original tensor and M is the total size (total number of
  ## elements) of the original tensor!
  ##
  ## Input:
  ##   - csvPath: a path to the csvfile
  ##   - skipHeader: should read_csv skip the first row
  ##   - separator: a char, default ','
  ##   - quote: a char, default '\"' (single and double quotes must be escaped).
  ##     Separators inside quoted strings are ignored, for example: `"foo", "bar, baz"` corresponds to 2 columns not 3.

  var parser: proc(x:string): T {.nimcall.}
  when T is SomeSignedInt:
    parser = proc(x: string): T = x.parseInt.T
  elif T is SomeUnsignedInt:
    parser = proc(x: string): T = x.parseUInt.T
  elif T is SomeFloat:
    parser = proc(x: string): T = x.parseFloat.T
  elif T is bool:
    parser = parseBool
  elif T is string:
    parser = proc(x: string): string =
      when defined(gcArc) or defined(gcOrc):
        result = x
      else:
        shallowCopy(result, x) # no-op

  # 1. count number of lines and columns using memfile interface
  let (numRows, numCols) = countLinesAndCols(csvPath, separator, quote, skipHeader)

  # 2. prepare CSV parser
  var csv: CsvParser
  let stream = newFileStream(csvPath, mode = fmRead)
  csv.open( stream, csvPath,
            separator = separator,
            quote = quote,
            skipInitialSpace = true
          )
  defer: csv.close

  # 3. possibly skip the header
  if skipHeader:
    csv.readHeaderRow()

  # 4. init data storage for each type & process all rows
  result = newTensorUninit[T]([numRows, numCols])
  var curRow = 0
  while csv.readRow:
    for i, val in csv.row:
      result[curRow, i] = parser val
    inc curRow

proc to_csv*[T](
    tensor: Tensor[T],
    csvPath: string,
    separator = ',',
    ) =
  ## Stores a tensor in a csv file. Can handle tensors of arbitrary dimension
  ## by using a schema (= csv columns) of
  ##
  ## dimension_1, dimension_2, ..., dimension_(tensor.rank), value
  ##
  ## where the 'dimension_i' columns contain indices, and the actual tensor
  ## values are stored in the 'value' column.
  ##
  ## For example the tensor ``@[@[1, 2, 3], @[4, 5, 6]].toTensor()`` is stored as:
  ##
  ## dimension_1,dimension_2,value
  ## 0,0,1
  ## 0,1,2
  ## 0,2,3
  ## 1,0,4
  ## 1,1,5
  ## 1,2,6
  ##
  ## Input:
  ##   - tensor: the tensor to store
  ##   - csvPath: output path of the csvfile
  ##   - separator: a char, default ','

  let file = open(csvPath, fmWrite)
  defer: file.close

  # write header
  for i in 1 .. tensor.rank:
    file.write("dimension_" & $i & separator)
  file.write("value\n")

  # sort dimensions by their strides (required for 'increment indices' below)
  var stride_order =
    toSeq(0 ..< tensor.rank).map(proc(i: int): (int, int) = (tensor.strides[i], i))
                            .sorted(system.cmp)
                            .map(proc(t: (int, int)): int = t[1])

  var indices = newSeq[int](tensor.rank)

  for i in 0 ..< tensor.size():
    # determine contiguous index
    var index = tensor.offset
    for dim in 0 ..< tensor.rank:
      index += tensor.strides[dim] * indices[dim]
    let value = tensor.atContiguousIndex(index)

    # write indices + values to file
    for i in 0 ..< tensor.rank:
      file.write($indices[i] & separator)
    file.write($value & "\n")

    # increment indices
    var j = 0
    while j < tensor.rank:
      # determine the index to increment depending on stride order
      let index_to_increment = stride_order[j]
      indices[index_to_increment] += 1
      # check if index hits the shape and propagate increment
      if indices[index_to_increment] == tensor.shape[index_to_increment]:
        indices[index_to_increment] = 0
        j += 1
      else:
        break
