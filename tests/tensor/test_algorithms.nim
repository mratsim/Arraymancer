# Copyright 2020 the Arraymancer contributors
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

suite "[Core] Testing algorithm functions":
  test "Sort":
    block:
      var t = @[4, 2, 7, 3, 1].toTensor()
      t.sort()
      check t == @[1, 2, 3, 4, 7].toTensor()

    block:
      var t = @[4, 2, 7, 3, 1].toTensor()
      t.sort(order = SortOrder.Descending)
      check t == @[7, 4, 3, 2, 1].toTensor()

  test "Sorted":
    block:
      var t = @[4, 2, 7, 3, 1].toTensor()
      check t.sorted == @[1, 2, 3, 4, 7].toTensor()

    block:
      var t = @[4, 2, 7, 3, 1].toTensor()
      check t.sorted(order = SortOrder.Descending) == @[7, 4, 3, 2, 1].toTensor()
