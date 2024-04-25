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
import std / unittest

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

  test "Argsort":
    block:
      let t = @[4, 2, 7, 3, 1].toTensor()
      let exp = @[4, 1, 3, 0, 2].toTensor()
      let idxSorted = t.argsort()
      check idxSorted == exp
      check t[idxSorted] == @[1, 2, 3, 4, 7].toTensor()

    block:
      let t = @[4, 2, 7, 3, 1].toTensor()
      let exp = @[2, 0, 3, 1, 4].toTensor()
      let idxSorted = t.argsort(order = SortOrder.Descending)
      check idxSorted == exp
      check t[idxSorted] == @[7, 4, 3, 2, 1].toTensor()

  test "Unique":
    block:
      let
        dup = [1, 3, 2, 4, 1, 8, 2, 1, 4].toTensor
        unique_unsorted = dup.unique
        unique_presorted_ascending = sorted(dup.unique).unique(isSorted = true)
        unique_presorted_descending = sorted(dup.unique, order = SortOrder.Descending).unique(isSorted = true)
        unique_sorted_ascending = dup.unique(order = SortOrder.Ascending)
        unique_sorted_descending = dup.unique(order = SortOrder.Descending)
        dup_not_C_continuous = dup[_ | 2]
        unique_not_c_continuous = dup_not_C_continuous.unique
        unique_sorted_not_c_continuous = dup_not_C_continuous.unique(order = SortOrder.Descending)

      check unique_unsorted == [1, 3, 2, 4, 8].toTensor
      check unique_presorted_ascending == [1, 2, 3, 4, 8].toTensor
      check unique_presorted_descending == [8, 4, 3, 2, 1].toTensor
      check unique_sorted_ascending == [1, 2, 3, 4, 8].toTensor
      check unique_sorted_descending == [8, 4, 3, 2, 1].toTensor
      check unique_not_c_continuous == [1, 2, 4].toTensor
      check unique_sorted_not_c_continuous == [4, 2, 1].toTensor

  test "Union":
    block:
      let t1 = [3, 1, 3, 2, 1, 0].toTensor
      let t2 = [4, 2, 2, 3].toTensor
      check: sorted(union(t1, t2)) == [0, 1, 2, 3, 4].toTensor

  test "Intersection":
    block:
      let t1 = [3, 1, 3, 2, 1, 0].toTensor
      let t2 = [4, 2, 2, 3].toTensor
      check: sorted(intersection(t1, t2)) == [2, 3].toTensor

