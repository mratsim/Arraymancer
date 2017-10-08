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

import  ./private/p_accessors,
        ./private/p_checks,
        ./data_structure

iterator items*[T](t: Tensor[T]): T {.inline,noSideEffect.} =
  ## Inline iterator on Tensor values
  ##
  ## The iterator will iterate in C order regardingless of the tensor properties (Fortran layout, non-contiguous, slice ...).
  ## So [0, 0, 0] then [0, 0, 1] then ... then [0, 1, 0] ...
  ##
  ## Usage:
  ##  .. code:: nim
  ##     for val in t: # items is implicitly called
  ##       val += 42
  stridedIteration(IterKind.Values, t, 0, t.size)

iterator items*[T](t: Tensor[T], offset, size: int): T {.inline,noSideEffect.} =
  ## Inline iterator on Tensor values (with offset)
  stridedIteration(IterKind.Values, t, offset, size)

iterator mitems*[T](t: var Tensor[T]): var T {.inline,noSideEffect.} =
  ## Inline iterator on Tensor values (mutable, with offset)
  stridedIteration(IterKind.Values, t, 0, t.size)

iterator mitems*[T](t: var Tensor[T], offset, size: int): var T {.inline,noSideEffect.} =
  ## Inline iterator on Tensor values (mutable, with offset)
  stridedIteration(IterKind.Values, t, offset, size)

iterator enumerate*[T](t: Tensor[T]): (int, T) {.inline.} =
  ## Enumerate Tensor values
  stridedIteration(IterKind.Iter_Values, t, 0, t.size)

iterator enumerate*[T](t: Tensor[T], offset, size: int): (int, T) {.inline,noSideEffect.} =
  ## Enumerate Tensor values (with offset)
  stridedIteration(IterKind.Iter_Values, t, offset, size)

iterator menumerate*[T](t: Tensor[T]): (int, var T) {.inline,noSideEffect.} =
  ## Enumerate Tensor values (mutable)
  stridedIteration(IterKind.Iter_Values, t, 0, t.size)

iterator menumerate*[T](t: Tensor[T], offset, size: int): (int, var T) {.inline,noSideEffect.} =
  ## Enumerate Tensor values (mutable, with offset)
  stridedIteration(IterKind.Iter_Values, t, offset, size)

iterator pairs*[T](t: Tensor[T]): (seq[int], T) {.inline,noSideEffect.} =
  ## Inline iterator on Tensor (coordinates, values)
  ##
  ## The iterator will iterate in C order regardingless of the tensor properties (Fortran layout, non-contiguous, slice ...).
  ## So [0, 0, 0] then [0, 0, 1] then ... then [0, 1, 0] ...
  ##
  ## It returns a tuple of (coordinates, value) like (@[1,0,1], 1337)
  ##
  ## Usage:
  ##  .. code:: nim
  ##     for coord, val in t:
  ##       echo coord
  ##       echo val
  ##  .. code:: nim
  ##     for coordval in t:
  ##       echo coordval[0]
  ##       echo coordval[1]
  stridedCoordsIteration(t, 0, t.size)

iterator mpairs*[T](t:var  Tensor[T]): (seq[int], var T) {.inline,noSideEffect.} =
  ## Inline iterator on Tensor (coordinates, values) (mutable)
  stridedCoordsIteration(t, 0, t.size)

iterator zip*[T,U](t1: Tensor[T], t2: Tensor[U]): (T,U) {.inline,noSideEffect.} =
  ## Iterates simultaneously on two tensors returning their elements in a tuple.
  ## Note: only tensors of the same shape will be zipped together.
  when compileOption("boundChecks"):
    check_size(t1, t2)
  dualStridedIteration(IterKind.Values, t1, t2, 0, t1.size)

iterator zip*[T,U](t1: Tensor[T], t2: Tensor[U], offset, size: int): (T,U) {.inline,noSideEffect.} =
  ## Iterates simultaneously on two tensors returning their elements in a tuple. (with offset)
  ## Note: only tensors of the same shape will be zipped together.
  when compileOption("boundChecks"):
    check_size(t1, t2)
  dualStridedIteration(IterKind.Values, t1, t2, offset, size)

iterator mzip*[T,U](t1: var Tensor[T], t2: Tensor[U]): (var T, U) {.inline,noSideEffect.} =
  ## Iterates simultaneously on two tensors returning their elements in a tuple. (mutable)
  ## Note: only tensors of the same shape will be zipped together.
  when compileOption("boundChecks"):
    check_size(t1, t2)
  dualStridedIteration(IterKind.Values, t1, t2, 0, t1.size)

iterator mzip*[T,U](t1: var Tensor[T], t2: Tensor[U], offset, size: int): (var T, U) {.inline,noSideEffect.} =
  ## Iterates simultaneously on two tensors returning their elements in a tuple. (mutable, with offset)
  ## Note: only tensors of the same shape will be zipped together.
  when compileOption("boundChecks"):
    check_size(t1, t2)
  dualStridedIteration(IterKind.Values, t1, t2, offset, size)

iterator enumerateZip*[T,U](t1: Tensor[T], t2: Tensor[U]): (int,T,U) {.inline,noSideEffect.} =
  ## Enumerate simultaneously on two tensors returning their elements in a tuple.
  ## Note: only tensors of the same shape will be zipped together.
  when compileOption("boundChecks"):
    check_size(t1, t2)
  dualStridedIteration(IterKind.Iter_Values, t1, t2, 0, t1.size)

iterator enumerateZip*[T,U](t1: Tensor[T], t2: Tensor[U], offset, size: int): (int,T,U) {.inline,noSideEffect.} =
  ## Enumerate simultaneously on two tensors returning their elements in a tuple. (with offset)
  ## Note: only tensors of the same shape will be zipped together.
  when compileOption("boundChecks"):
    check_size(t1, t2)
  dualStridedIteration(IterKind.Iter_Values, t1, t2, offset, size)

iterator menumerateZip*[T,U](t1: var Tensor[T], t2: Tensor[U]): (int, var T,U) {.inline,noSideEffect.} =
  ## Enumerate simultaneously on two tensors returning their elements in a tuple. (mutable)
  ## Note: only tensors of the same shape will be zipped together.
  when compileOption("boundChecks"):
    check_size(t1, t2)
  dualStridedIteration(IterKind.Iter_Values, t1, t2, 0, t1.size)

iterator menumerateZip*[T,U](t1: var Tensor[T], t2: Tensor[U], offset, size: int): (int, var T,U) {.inline,noSideEffect.} =
  ## Enumerate simultaneously on two tensors returning their elements in a tuple. (mutable, with offset)
  ## Note: only tensors of the same shape will be zipped together.
  when compileOption("boundChecks"):
    check_size(t1, t2)
  dualStridedIteration(IterKind.Iter_Values, t1, t2, offset, size)

template axis_iterator[T](t: Tensor[T], axis: int): untyped =
  var out_t = t

  let axis_len = t.shape[axis]
  let axis_stride = t.strides[axis]
  out_t.shape[axis] = 1

  for _ in 0..<axis_len:
    yield out_t
    out_t.offset += axis_stride

iterator axis*[T](t: Tensor[T], axis: int): Tensor[T] {.inline,noSideEffect.}=
  ## Inline iterator over an axis.
  ##
  ## Returns:
  ##   - A slice along the given axis at each iteration.
  ##
  ## Note: The slice dimension is not collapsed by default.
  ## You can use ``unsafeSqueeze`` to collapse it without copy.
  ## In this case ``unsafeSqueeze`` is safe.
  ##
  ## Usage:
  ##  .. code:: nim
  ##     for subtensor in t.axis(1):
  ##       # do stuff
  axis_iterator(t,axis)

proc axis*[T](t: Tensor[T], axis: int): auto {.noSideEffect.}=
  ## Closure iterator on axis
  ##
  ## A closure iterator must be assigned to an iterator variable first.
  ## Usage:
  ##  .. code:: nim
  ##     let it = t.axis(1)
  ##     for subtensor in it():
  ##       # do stuff
  ##
  ## Note: This is mostly useful for iterator chaining.
  ## Prefer the inline iterator ``axis`` for simple iteration.
  return iterator(): Tensor[T] = axis_iterator(t,axis)
