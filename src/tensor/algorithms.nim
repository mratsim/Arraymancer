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

import ./data_structure

import std / algorithm
export SortOrder

proc sort*[T](t: var Tensor[T], order = SortOrder.Ascending) =
  ## Sorts the given tensor inplace. For the time being this is only supported for
  ## 1D tensors!
  assert t.rank == 1, "Only 1D tensors can be sorted at the moment!"
  sort(toOpenArray(t.storage.Fdata, 0, t.size - 1), order = order)

proc sorted*[T](t: Tensor[T], order = SortOrder.Ascending): Tensor[T] =
  ## Returns a sorted version of the given tensor `t`. Also only supported for
  ## 1D tensors for the time being!
  result = t.clone
  result.sort(order = order)
