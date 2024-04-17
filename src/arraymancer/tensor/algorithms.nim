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
  var mt = t.toUnsafeView # without this we get an error that the openArray is immutable?
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
    mt = t.clone.toUnsafeView # without this we get an error that the openArray is immutable?
  else:
    mt = t.toUnsafeView # without this we get an error that the openArray is immutable?
  var tups = zip(toOpenArray(mt, 0, t.size - 1),
                 toSeq(0 ..< t.size))
  # sort by custom sort proc
  tups.sort(cmp = cmpIdxTup, order = order)
  result = newTensorUninit[int](t.shape)
  for i in 0 ..< t.size:
    result[i] = tups[i][1]

proc unique*[T](t: Tensor[T], isSorted=false): Tensor[T] =
  ## Return a new Tensor with the unique elements of the input Tensor in the order they first appear
  ##
  ## Note that this is the *"unsorted"* version of this procedure which returns
  ## the unique values in the order in which they first appear on the input.
  ## Do not get confused by the `isSorted` argument which is not used to sort
  ## the output, but to make the algorithm more efficient when the input tensor
  ## is already sorted.
  ##
  ## There is another version of this procedure which gets an `order` argument
  ## that let's you sort the output (in ascending or descending order).
  ##
  ## Inputs:
  ##   - t: The input Tensor
  ##   - isSorted: Set this to `true` if the input tensor is already sorted,
  ##               in order to use a more efficient algorithm for finding the
  ##               unique of the input Tensor. Be careful however when using
  ##               this option, since if the input tensor is not really sorted,
  ##               the output will be wrong.
  ##
  ## Result:
  ##   - A new Tensor with the unique elements of the input Tensor in the order
  ##     in which they first appear on the input Tensor.
  ##
  ## Examples:
  ## ```nim
  ## let
  ##   dup = [1, 3, 2, 4, 1, 8, 2, 1, 4].toTensor
  ## assert dup.unique == [1, 3, 2, 4, 8].toTensor
  ##
  ## # Use `isSorted = true` only if the input tensor is already sorted
  ## assert dup.sorted.unique(isSorted = true) == [1, 3, 2, 4, 8].toTensor
  ## ```

  if t.is_C_contiguous:
    # Note that since deduplicate returns a new sequence, it is safe to apply it
    # to a view of the raw data of the input tensor
    toOpenArray(t.toUnsafeView, 0, t.size - 1).deduplicate(isSorted = isSorted).toTensor
  else:
    # Clone the tensor in order to make it C continuous and then make it unique
    unique(t.clone(), isSorted = isSorted)

proc unique*[T](t: Tensor[T], order: SortOrder): Tensor[T] =
  ## Return a new sorted Tensor with the unique elements of the input Tensor
  ##
  ## Note that this is the "sorted" version of this procedure. There is
  ## another version which doesn't get a `sort` argument that returns the
  ## unique elements int he order in which they first appear ont he input.
  ##
  ## Inputs:
  ##   - t: The input Tensor
  ##   - order: The order in which elements are sorted (`SortOrder.Ascending` or `SortOrder.Descending`)
  ##
  ## Result:
  ##   - A new Tensor with the unique elements of the input Tensor sorted in the specified order.
  ##
  ## Examples:
  ## ```nim
  ## let
  ##   dup = [1, 3, 2, 4, 1, 8, 2, 1, 4].toTensor
  ##   unique_ascending_sort = dup.unique(order = SortOrder.Ascending)
  ##   unique_descending_sort = dup.unique(order = SortOrder.Descending)
  ## assert unique_ascending_sort == [1, 2, 3, 4, 8].toTensor
  ## assert unique_descending_sort == [8, 4, 3, 2, 1].toTensor
  ## ```

  if t.is_C_contiguous:
    # Note that since sorted returns a new sequence, it is safe to apply it
    # to a view of the raw data of the input tensor
    sorted(toOpenArray(t.toUnsafeView, 0, t.size - 1),
        order = order)
        .deduplicate(isSorted = true).toTensor
  else:
    # We need to clone the tensor in order to make it C continuous
    # and then we can make it unique assuming that it is already sorted
    sorted(t, order = order).unique(isSorted = true)
