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

import ./data_structure,
       ./init_cpu,
       ./init_copy_cpu

import std / [algorithm, sequtils]
export SortOrder

proc sort*[T](t: var Tensor[T], order = SortOrder.Ascending) =
  ## Sorts the given tensor inplace. For the time being this is only supported for
  ## 1D tensors!
  ##
  ## Sorts the raw underlying data!
  # TODO: if `t` is a view, this will sort everything
  assert t.rank == 1, "Only 1D tensors can be sorted at the moment!"
  var mt = t.dataArray # without this we get an error that the openArray is immutable?
  sort(toOpenArray(mt, 0, t.size - 1), order = order)

proc sorted*[T](t: Tensor[T], order = SortOrder.Ascending): Tensor[T] =
  ## Returns a sorted version of the given tensor `t`. Also only supported for
  ## 1D tensors for the time being!
  result = t.clone
  result.sort(order = order)

proc argsort*[T](t: Tensor[T], order = SortOrder.Ascending, toCopy = false): Tensor[int] =
  ## Returns the indices which would sort `t`. Useful to apply the same sorting to
  ## multiple tensors based on the order of the tensor `t`.
  ##
  ## If `toCopy` is `true` the input tensor is cloned. Else it is already sorted.
  # TODO: should we clone `t` so that if `t` is a view we don't access the whole
  # data?
  assert t.rank == 1, "Only 1D tensors can be sorted at the moment!"
  proc cmpIdxTup(x, y: (T, int)): int = system.cmp(x[0], y[0])
  # make a tuple of input & indices
  var mt: ptr UncheckedArray[T]
  if toCopy:
    mt = t.clone.dataArray # without this we get an error that the openArray is immutable?
  else:
    mt = t.dataArray # without this we get an error that the openArray is immutable?
  var tups = zip(toOpenArray(mt, 0, t.size - 1),
                 toSeq(0 ..< t.size))
  # sort by custom sort proc
  tups.sort(cmp = cmpIdxTup, order = order)
  result = newTensorUninit[int](t.shape)
  for i in 0 ..< t.size:
    result[i] = tups[i][1]
