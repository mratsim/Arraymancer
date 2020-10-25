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

proc atContiguousIndex*[T](t: Tensor[T], idx: int): T {.noSideEffect,inline.} =
  ## Return value of tensor at contiguous index
  ## i.e. as treat the tensor as flattened
  return t.unsafe_raw_buf[t.getContiguousIndex(idx)]

proc atContiguousIndex*[T](t: var Tensor[T], idx: int): var T {.noSideEffect,inline.} =
  ## Return value of tensor at contiguous index (mutable)
  ## i.e. as treat the tensor as flattened
  return t.unsafe_raw_buf[t.getContiguousIndex(idx)]

proc atAxisIndex*[T](t: Tensor[T], axis, idx: int, length = 1): Tensor[T] {.noInit,inline.} =
  ## Returns a sliced tensor in the given axis index

  when compileOption("boundChecks"):
    check_axis_index(t, axis, idx, length)

  result = t
  result.shape[axis] = length
  result.offset += result.strides[axis]*idx

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
  when compileOption("boundChecks"):
    check_contiguous_index(t, offset)
    check_contiguous_index(t, offset+size-1)
  stridedIteration(IterKind.Values, t, offset, size)

iterator mitems*[T](t: var Tensor[T]): var T {.inline,noSideEffect.} =
  ## Inline iterator on Tensor values (mutable, with offset)
  ##
  ## Note: due to C++ restrictions and Nim current codegen on mutable iterator,
  ## it is not possible to use this iterator with the C++ backend
  ## or at the same time as Cuda (that uses C++)
  stridedIteration(IterKind.Values, t, 0, t.size)

iterator mitems*[T](t: var Tensor[T], offset, size: int): var T {.inline,noSideEffect.} =
  ## Inline iterator on Tensor values (mutable, with offset)
  ##
  ## Note: due to C++ restrictions and Nim current codegen on mutable iterator,
  ## it is not possible to use this iterator with the C++ backend
  ## or at the same time as Cuda (that uses C++)
  when compileOption("boundChecks"):
    check_contiguous_index(t, offset)
    check_contiguous_index(t, offset+size-1)
  stridedIteration(IterKind.Values, t, offset, size)

iterator enumerate*[T](t: Tensor[T]): (int, T) {.inline.} =
  ## Enumerate Tensor values
  stridedIteration(IterKind.Iter_Values, t, 0, t.size)

iterator enumerate*[T](t: Tensor[T], offset, size: int): (int, T) {.inline,noSideEffect.} =
  ## Enumerate Tensor values (with offset)
  when compileOption("boundChecks"):
    check_contiguous_index(t, offset)
    check_contiguous_index(t, offset+size-1)
  stridedIteration(IterKind.Iter_Values, t, offset, size)

iterator menumerate*[T](t: Tensor[T]): (int, var T) {.inline,noSideEffect.} =
  ## Enumerate Tensor values (mutable)
  ##
  ## Note: due to C++ restrictions and Nim current codegen on mutable iterator,
  ## it is not possible to use this iterator with the C++ backend
  ## or at the same time as Cuda (that uses C++)
  stridedIteration(IterKind.Iter_Values, t, 0, t.size)

iterator menumerate*[T](t: Tensor[T], offset, size: int): (int, var T) {.inline,noSideEffect.} =
  ## Enumerate Tensor values (mutable, with offset)
  ##
  ## Note: due to C++ restrictions and Nim current codegen on mutable iterator,
  ## it is not possible to use this iterator with the C++ backend
  ## or at the same time as Cuda (that uses C++)
  when compileOption("boundChecks"):
    check_contiguous_index(t, offset)
    check_contiguous_index(t, offset+size-1)
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
  ##
  ## Note: due to C++ restrictions and Nim current codegen on mutable iterator,
  ## it is not possible to use this iterator with the C++ backend
  ## or at the same time as Cuda (that uses C++)
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
    check_contiguous_index(t1, offset)
    check_contiguous_index(t1, offset+size-1)
  dualStridedIteration(IterKind.Values, t1, t2, offset, size)

iterator zip*[T,U,V](t1: Tensor[T], t2: Tensor[U], t3: Tensor[V]): (T,U,V) {.inline,noSideEffect.} =
  ## Iterates simultaneously on two tensors returning their elements in a tuple.
  ## Note: only tensors of the same shape will be zipped together.
  when compileOption("boundChecks"):
    check_size(t1, t2)
    check_size(t1, t3)
  tripleStridedIteration(IterKind.Values, t1, t2, t3, 0, t1.size)

iterator zip*[T,U,V](t1: Tensor[T], t2: Tensor[U], t3: Tensor[V], offset, size: int): (T,U,V) {.inline,noSideEffect.} =
  ## Iterates simultaneously on two tensors returning their elements in a tuple. (with offset)
  ## Note: only tensors of the same shape will be zipped together.
  when compileOption("boundChecks"):
    check_size(t1, t2)
    check_size(t1, t3)
    check_contiguous_index(t1, offset)
    check_contiguous_index(t1, offset+size-1)
  tripleStridedIteration(IterKind.Values, t1, t2, t3, offset, size)

iterator mzip*[T,U](t1: var Tensor[T], t2: Tensor[U]): (var T, U) {.inline,noSideEffect.} =
  ## Iterates simultaneously on two tensors returning their elements in a tuple. (mutable)
  ## Note: only tensors of the same shape will be zipped together.
  ##
  ## Note: due to C++ restrictions and Nim current codegen on mutable iterator,
  ## it is not possible to use this iterator with the C++ backend
  ## or at the same time as Cuda (that uses C++)
  when compileOption("boundChecks"):
    check_size(t1, t2)
  dualStridedIteration(IterKind.Values, t1, t2, 0, t1.size)

iterator mzip*[T,U](t1: var Tensor[T], t2: Tensor[U], offset, size: int): (var T, U) {.inline,noSideEffect.} =
  ## Iterates simultaneously on two tensors returning their elements in a tuple. (mutable, with offset)
  ## Note: only tensors of the same shape will be zipped together.
  ##
  ## Note: due to C++ restrictions and Nim current codegen on mutable iterator,
  ## it is not possible to use this iterator with the C++ backend
  ## or at the same time as Cuda (that uses C++)
  when compileOption("boundChecks"):
    check_size(t1, t2)
    check_contiguous_index(t1, offset)
    check_contiguous_index(t1, offset+size-1)
  dualStridedIteration(IterKind.Values, t1, t2, offset, size)

iterator mzip*[T,U,V](t1: var Tensor[T], t2: Tensor[U], t3: Tensor[V]): (var T, U, V) {.inline,noSideEffect.} =
  ## Iterates simultaneously on two tensors returning their elements in a tuple. (mutable)
  ## Note: only tensors of the same shape will be zipped together.
  ##
  ## Note: due to C++ restrictions and Nim current codegen on mutable iterator,
  ## it is not possible to use this iterator with the C++ backend
  ## or at the same time as Cuda (that uses C++)
  when compileOption("boundChecks"):
    check_size(t1, t2)
  tripleStridedIteration(IterKind.Values, t1, t2, t3, 0, t1.size)

iterator mzip*[T,U,V](t1: var Tensor[T], t2: Tensor[U], t3: Tensor[V], offset, size: int): (var T, U, V) {.inline,noSideEffect.} =
  ## Iterates simultaneously on two tensors returning their elements in a tuple. (mutable, with offset)
  ## Note: only tensors of the same shape will be zipped together.
  ##
  ## Note: due to C++ restrictions and Nim current codegen on mutable iterator,
  ## it is not possible to use this iterator with the C++ backend
  ## or at the same time as Cuda (that uses C++)
  when compileOption("boundChecks"):
    check_size(t1, t2)
    check_size(t1, t3)
    check_contiguous_index(t1, offset)
    check_contiguous_index(t1, offset+size-1)
  tripleStridedIteration(IterKind.Values, t1, t2, t3, offset, size)

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
    check_contiguous_index(t1, offset)
    check_contiguous_index(t1, offset+size-1)
  dualStridedIteration(IterKind.Iter_Values, t1, t2, offset, size)

iterator enumerateZip*[T,U,V](t1: Tensor[T], t2: Tensor[U], t3: Tensor[V]): (int,T,U,V) {.inline,noSideEffect.} =
  ## Enumerate simultaneously on two tensors returning their elements in a tuple.
  ## Note: only tensors of the same shape will be zipped together.
  when compileOption("boundChecks"):
    check_size(t1, t2)
    check_size(t1, t3)
  tripleStridedIteration(IterKind.Iter_Values, t1, t2, t3, 0, t1.size)

iterator enumerateZip*[T,U,V](t1: Tensor[T], t2: Tensor[U], t3: Tensor[V], offset, size: int): (int,T,U,V) {.inline,noSideEffect.} =
  ## Enumerate simultaneously on two tensors returning their elements in a tuple. (with offset)
  ## Note: only tensors of the same shape will be zipped together.
  when compileOption("boundChecks"):
    check_size(t1, t2)
    check_size(t1, t3)
    check_contiguous_index(t1, offset)
    check_contiguous_index(t1, offset+size-1)
  tripleStridedIteration(IterKind.Iter_Values, t1, t2, t3, offset, size)

iterator menumerateZip*[T,U](t1: var Tensor[T], t2: Tensor[U]): (int, var T,U) {.inline,noSideEffect.} =
  ## Enumerate simultaneously on two tensors returning their elements in a tuple. (mutable)
  ## Note: only tensors of the same shape will be zipped together.
  ##
  ## Note: due to C++ restrictions and Nim current codegen on mutable iterator,
  ## it is not possible to use this iterator with the C++ backend
  ## or at the same time as Cuda (that uses C++)
  when compileOption("boundChecks"):
    check_size(t1, t2)
  dualStridedIteration(IterKind.Iter_Values, t1, t2, 0, t1.size)

iterator menumerateZip*[T,U](t1: var Tensor[T], t2: Tensor[U], offset, size: int): (int, var T,U) {.inline,noSideEffect.} =
  ## Enumerate simultaneously on two tensors returning their elements in a tuple. (mutable, with offset)
  ## Note: only tensors of the same shape will be zipped together.
  ##
  ## Note: due to C++ restrictions and Nim current codegen on mutable iterator,
  ## it is not possible to use this iterator with the C++ backend
  ## or at the same time as Cuda (that uses C++)
  when compileOption("boundChecks"):
    check_size(t1, t2)
    check_contiguous_index(t1, offset)
    check_contiguous_index(t1, offset+size-1)
  dualStridedIteration(IterKind.Iter_Values, t1, t2, offset, size)

template axisIterator[T](t: Tensor[T], axis, iter_offset, iter_size: int): untyped =
  when compileOption("boundChecks"):
    check_axis_index(t, axis, iter_offset, iter_size-1)

  var out_t = t.atAxisIndex(axis, iter_offset)

  for _ in 0..<iter_size:
    yield out_t
    out_t.offset += t.strides[axis]

template dualAxisIterator[T, U](a: Tensor[T], b: Tensor[U], axis, iter_offset, iter_size: int): untyped =
  when compileOption("boundChecks"):
    check_axis_index(a, axis, iter_offset, iter_size-1)
    assert a.shape[axis] == b.shape[axis] # TODO use a proper check

  var out_a = a.atAxisIndex(axis, iter_offset)
  var out_b = b.atAxisIndex(axis, iter_offset)

  for _ in 0..<iter_size:
    yield (out_a, out_b)
    out_a.offset += a.strides[axis]
    out_b.offset += b.strides[axis]

iterator axis*[T](t: Tensor[T], axis: int): Tensor[T] {.inline.}=
  ## Inline iterator over an axis.
  ##
  ## Returns:
  ##   - A slice along the given axis at each iteration.
  ##
  ## Note: The slice dimension is not collapsed by default.
  ## You can use ``squeeze`` to collapse it.
  ##
  ## Usage:
  ##  .. code:: nim
  ##     for subtensor in t.axis(1):
  ##       # do stuff
  axisIterator(t, axis, 0, t.shape[axis])

iterator axis*[T](t: Tensor[T], axis, offset, size: int): Tensor[T] {.inline.}=
  axisIterator(t, axis, offset, size)

iterator zipAxis*[T, U](a: Tensor[T], b: Tensor[U], axis: int): tuple[a: Tensor[T], b: Tensor[U]] {.inline.}=
  ## Inline iterator over 2 tensors over an axis.
  ##
  ## Returns:
  ##   - 2 slices along the given axis at each iteration.
  ##
  ## Note: The slice dimension is not collapsed by default.
  ## You can use ``squeeze`` to collapse it.
  ##
  ## Usage:
  ##  .. code:: nim
  ##     for subtensor in axis(a, b, 1):
  ##       # do stuff
  dualAxisIterator(a, b, axis, 0, a.shape[axis])

template enumerateAxisIterator[T](t: Tensor[T], axis, iter_offset, iter_size: int): untyped =
  var out_t = t.atAxisIndex(axis, iter_offset) # do we need a clone there?

  for i in 0..<iter_size:
    yield (i + iter_offset, out_t)
    out_t.offset += t.strides[axis]

iterator enumerateAxis*[T](t: Tensor[T], axis: int): (int, Tensor[T]) {.inline.}=
  ## Inline iterator over an axis.
  ##
  ## Returns a tuple:
  ##   - The index along the axis
  ##   - A slice along the given axis at each iteration.
  ##
  ## Note: The slice dimension is not collapsed by default.
  ## You can use ``squeeze`` to collapse it.
  ##
  ## Usage:
  ##  .. code:: nim
  ##     for subtensor in t.axis(1):
  ##       # do stuff
  enumerateAxisIterator(t, axis, 0, t.shape[axis])

iterator enumerateAxis*[T](t: Tensor[T], axis, offset, size: int): (int, Tensor[T]) {.inline.}=
  enumerateAxisIterator(t, axis, offset, size)
