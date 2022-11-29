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

import ../../src/arraymancer
import unittest

proc main() =
  suite "Optimization":
    test "Test if contiguous slices are detected as contiguous":
      let a = [[1, 2, 3, 4, 5],
              [6, 7, 8, 9, 10]].toTensor

      check: a[1, 2..3].isContiguous == true


main()
GC_fullCollect()
