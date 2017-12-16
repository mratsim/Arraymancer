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


import  os, parsecsv, streams, strutils,
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

  var parser: proc(x:string): T
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

  csv.close

  result = newTensorUninit[T](num_rows, num_cols)
  shallowCopy(result.storage.Fdata, csvdata)