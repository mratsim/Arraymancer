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

import ../../src/arraymancer, ../testutils
import unittest, os

proc main() =

  let expected_output_1d = """dimension_1,value
0,1
1,2
2,3
3,4
4,5
"""

  let expected_output_2d = """dimension_1,dimension_2,value
0,0,1
0,1,2
0,2,3
1,0,4
1,1,5
1,2,6
"""

  let expected_output_3d = """dimension_1,dimension_2,dimension_3,value
0,0,0,1
0,0,1,2
0,1,0,A
0,1,1,B
1,0,0,10
1,0,1,20
1,1,0,X
1,1,1,Y
"""

  let expected_output_semicolon = """dimension_1;dimension_2;value
0;0;1.0
0;1;2.0
1;0;3.0
1;1;4.0
"""

  let csv_empty_lines = """


dimension_1,value
0,1
1,2
2,3
3,4
4,5


"""

  let csv_semicolon_short = """dimension_1;value
0;1
1;2
2;3
3;4
4;5
"""

  let csv_with_quoted = """dimension_1,value
0,A
1,B
2,"hello, this is a string
with a line break, ugh
"
3,D
4,E
"""



  let test_file_path = getTempDir() / "arraymancer_test.csv"

  ## NOTE: the reading of CSV files in arraymancer currently ``does not`` handle parsing its own
  ## CSV files as the dimensional information becomes part of the CSV output. I.e. instead of constructing
  ## a NxMx...xZ tensor we always construct a NxM tensor, where N-1 is the rank of the original tensor
  ## and M is the total size (total number of elements) of the original.
  suite "[IO] CSV support":

    test "Should export 1d Tensor":
      let t = @[1, 2, 3, 4, 5].toTensor()
      t.to_csv(test_file_path)
      let content = readFile(test_file_path)
      check content == expected_output_1d

    test "Read 1D serialized tensor":
      let tRead = readCsv[int](test_file_path, skipHeader = true)
      let tExp = @[@[0, 1], @[1, 2], @[2, 3], @[3, 4], @[4, 5]].toTensor()
      check tExp == tRead

    test "Should export 2d Tensor":
      let t = @[@[1, 2, 3], @[4, 5, 6]].toTensor()
      t.to_csv(test_file_path)
      let content = readFile(test_file_path)
      check content == expected_output_2d

    test "Should export 3d Tensor":
      let t = @[@[@["1", "2"], @["A", "B"]], @[@["10", "20"], @["X", "Y"]]].toTensor()
      t.to_csv(test_file_path)
      let content = readFile(test_file_path)
      check content == expected_output_3d

    test "Should handle separator":
      let t = @[@[1.0, 2.0], @[3.0, 4.0]].toTensor()
      t.to_csv(test_file_path, separator=';')
      let content = readFile(test_file_path)
      check content == expected_output_semicolon

    test "CSV parsing ignores empty lines":
      writeFile(test_file_path, csv_empty_lines)
      let tRead = readCsv[int](test_file_path, skipHeader = true)
      let tExp = @[@[0, 1], @[1, 2], @[2, 3], @[3, 4], @[4, 5]].toTensor()
      check tExp == tRead

    test "CSV parsing of different (semicolon) separators works":
      writeFile(test_file_path, csv_semicolon_short)
      let tRead = readCsv[int](test_file_path, separator = ';', skipHeader = true)
      let tExp = @[@[0, 1], @[1, 2], @[2, 3], @[3, 4], @[4, 5]].toTensor()
      check tExp == tRead

    test "CSV parsing of file with quoted content works":
      writeFile(test_file_path, csv_with_quoted)
      let tRead = readCsv[string](test_file_path, quote = '\"', skipHeader = true)
      let tExp = @[@["0", "A"],
                   @["1", "B"],
                   @["2", """hello, this is a string
with a line break, ugh
"""],
                   @["3", "D"],
                   @["4", "E"]].toTensor()
      check tExp == tRead

main()
GC_fullCollect()
