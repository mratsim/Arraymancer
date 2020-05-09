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

let test_file_path = getTempDir() / "arraymancer_test.csv"

testSuite "[IO] CSV support":

  test "Should export 1d Tensor":
    let t = @[1, 2, 3, 4, 5].toTensor()
    t.to_csv(test_file_path)
    let content = readFile(test_file_path)
    check content == expected_output_1d

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
