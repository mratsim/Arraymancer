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
        ../tensor/tensor

proc read_csv*[T: SomeNumber|bool|string](
       csvPath: string,
       skip_header = false,
       separator = ',',
       quote = '\"'
       ): Tensor[T] {.noInit.} =
  ## Load a csv into a Tensor. All values must be of the same type.
  ##
  ## If there is a header row, it can be skipped.
  ##
  ## Input:
  ##   - csvPath: a path to the csvfile
  ##   - skip_header: should read_csv skip the first row
  ##   - separator: a char, default ','
  ##   - quote: a char, default '\"' (single and double quotes must be escaped).
  ##     Separators inside quoted strings are ignored, for example: `"foo", "bar, baz"` corresponds to 2 columns not 3.

  var parser: proc(x:string): T {.nimcall.}
  when T is SomeSignedInt:
    parser = proc(x:string): T = x.parseInt.T
  elif T is SomeUnsignedInt:
    parser = proc(x:string): T = x.parseUInt.T
  elif T is SomeReal:
    parser = proc(x:string): T = x.parseFloat.T
  elif T is bool:
    parser = parseBool
  elif T is string:
    parser = proc(x: string): string = shallowCopy(result, x) # no-op

  var csv: CsvParser
  let stream = newFileStream(csvPath, mode = fmRead)

  csv.open( stream, csvPath,
            separator = separator,
            quote = quote,
            skipInitialSpace = true
          )
  defer: csv.close

  if skip_header:
    discard csv.readRow

  # Initialization, count cols:
  discard csv.readRow #TODO what if there is only one line.
  var
    num_cols = csv.row.len
    csvdata: seq[T] = @[]
  for val in csv.row:
    csvdata.add parser(val)

  # Processing
  while csv.readRow:
    for val in csv.row:
      csvdata.add parser(val)

  # Finalizing
  let num_rows= if skip_header: csv.processedRows - 2
                else: csv.processedRows - 1

  result = newTensorUninit[T](num_rows, num_cols)
  shallowCopy(result.storage.Fdata, csvdata)


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
