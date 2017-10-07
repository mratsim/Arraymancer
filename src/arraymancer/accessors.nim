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


# ######################################################
# This file implements iterators to iterate on Tensors.

# ##############################################################
# The reference implementation below went through several optimizations:
#  - Using non-initialized stack allocation (array instead of seq)
#  - Avoiding closures in all higher-order functions, even when iterating on 2 tensors at the same time

# ###### Reference implementation ######

# template strided_iteration[T](t: Tensor[T], strider: IterKind): untyped =
#   ## Iterate over a Tensor, displaying data as in C order, whatever the strides.
#
#   ## Iterator init
#   var coord = newSeq[int](t.rank) # Coordinates in the n-dimentional space
#   var backstrides: seq[int] = @[] # Offset between end of dimension and beginning
#   for i,j in zip(t.strides,t.shape):
#     backstrides.add(i*(j-1))
#
#   var iter_pos = t.offset
#
#   ## Iterator loop
#   for i in 0 .. <t.shape.product:
#
#     ## Templating the return value
#     when strider == IterKind.Values: yield t.data[iter_pos]
#     elif strider == IterKind.Coord_Values: yield (coord, t.data[iter_pos])
#     elif strider == IterKind.MemOffset: yield iter_pos
#     elif strider == IterKind.MemOffset_Values: yield (iter_pos, t.data[iter_pos])
#
#     ## Computing the next position
#     for k in countdown(t.rank - 1,0):
#       if coord[k] < t.shape[k]-1:
#         coord[k] += 1
#         iter_pos += t.strides[k]
#         break
#       else:
#         coord[k] = 0
#         iter_pos -= backstrides[k]



proc check_index(t: Tensor, idx: varargs[int]) {.noSideEffect.}=
  if idx.len != t.rank:
    raise newException(IndexError, "Number of arguments: " &
                    $(idx.len) &
                    ", is different from tensor rank: " &
                    $(t.rank))

proc check_elementwise(a, b:AnyTensor)  {.noSideEffect.}=
  ## Check if element-wise operations can be applied to 2 Tensors
  if a.shape != b.shape:
    raise newException(ValueError, "Both Tensors should have the same shape.\n Left-hand side has shape " &
                                   $a.shape & " while right-hand side has shape " & $b.shape)

proc check_size[T,U](a:Tensor[T], b:Tensor[U])  {.noSideEffect.}=
  ## Check if the total number of elements match
  if a.size != b.size:
    raise newException(ValueError, "Both Tensors should have the same total number of elements.\n" &
      "Left-hand side has " & $a.size & " (shape: " & $a.shape & ") while right-hand side has " &
      $b.size & " (shape: " & $b.shape & ")."
    )

proc getIndex[T](t: Tensor[T], idx: varargs[int]): int {.noSideEffect,inline.} =
  ## Convert [i, j, k, l ...] to the proper index.
  when compileOption("boundChecks"):
    t.check_index(idx)

  var real_idx = t.offset
  for i in 0..<idx.len:
    real_idx += t.strides[i]*idx[i]
  return real_idx

proc atIndex[T](t: Tensor[T], idx: varargs[int]): T {.noSideEffect,inline.} =
  ## Get the value at input coordinates
  ## This used to be `[]` before slicing was implemented
  return t.data[t.getIndex(idx)]

proc atIndex[T](t: var Tensor[T], idx: varargs[int]): var T {.noSideEffect,inline.} =
  ## Get the value at input coordinates
  ## This allows inplace operators t[1,2] += 10 syntax
  return t.data[t.getIndex(idx)]

proc atIndexMut[T](t: var Tensor[T], idx: varargs[int], val: T) {.noSideEffect,inline.} =
  ## Set the value at input coordinates
  ## This used to be `[]=` before slicing was implemented
  t.data[t.getIndex(idx)] = val
## Iterators
type
  IterKind = enum
    Values, Iter_Values

template initStridedIteration(coord, backstrides, iter_pos: untyped, t, iter_offset, iter_size: typed): untyped =
  ## Iterator init
  var iter_pos = 0
  var coord {.noInit.}: array[MAXRANK, int]
  var backstrides {.noInit.}: array[MAXRANK, int]
  for i in 0..<t.rank:
    backstrides[i] = t.strides[i]*(t.shape[i]-1)
    coord[i] = 0

  # Calculate initial coords and iter_pos from iteration offset
  if iter_offset != 0:
    var z = 1
    for i in countdown(t.rank - 1,0):
      let z2 = z*t.shape[i]
      coord[i] = (iter_offset div z) mod z2
      iter_pos += coord[i]*t.strides[i]
      z = z2

template advanceStridedIteration(coord, backstrides, iter_pos, t, iter_offset, iter_size: typed): untyped =
  ## Computing the next position
  for k in countdown(t.rank - 1,0):
    if coord[k] < t.shape[k]-1:
      coord[k] += 1
      iter_pos += t.strides[k]
      break
    else:
      coord[k] = 0
      iter_pos -= backstrides[k]

template stridedIterationYield(strider: IterKind, data, i, iter_pos: typed) =
  ## Iterator the return value
  when strider == IterKind.Values: yield data[iter_pos]
  elif strider == IterKind.Iter_Values: yield (i, data[iter_pos])

template stridedIteration(strider: IterKind, t, iter_offset, iter_size: typed): untyped =
  ## Iterate over a Tensor, displaying data as in C order, whatever the strides.

  # Get tensor data address with offset builtin
  var data = t.dataArray

  # Optimize for loops in contiguous cases
  if t.is_C_Contiguous:
    for i in iter_offset..<(iter_offset+iter_size):
      stridedIterationYield(strider, data, i, i)
  else:
    initStridedIteration(coord, backstrides, iter_pos, t, iter_offset, iter_size)
    for i in iter_offset..<(iter_offset+iter_size):
      stridedIterationYield(strider, data, i, iter_pos)
      advanceStridedIteration(coord, backstrides, iter_pos, t, iter_offset, iter_size)

template stridedCoordsIteration(t, iter_offset, iter_size: typed): untyped =
  ## Iterate over a Tensor, displaying data as in C order, whatever the strides. (coords)

  # Get tensor data address with offset builtin
  var data = t.dataArray
  let rank = t.rank
  initStridedIteration(coord, backstrides, iter_pos, t, iter_offset, iter_size)
  for i in iter_offset..<(iter_offset+iter_size):
    yield (coord[0..<rank], data[iter_pos])
    advanceStridedIteration(coord, backstrides, iter_pos, t, iter_offset, iter_size)

template dualStridedIterationYield(strider: IterKind, t1data, t2data, i, t1_iter_pos, t2_iter_pos: typed) =
  ## Iterator the return value
  when strider == IterKind.Values: yield (t1data[t1_iter_pos], t2data[t2_iter_pos])
  elif strider == IterKind.Iter_Values: yield (i, t1data[t1_iter_pos], t2data[t2_iter_pos])

template dualStridedIteration(strider: IterKind, t1, t2, iter_offset, iter_size: typed): untyped =
  ## Iterate over two Tensors, displaying data as in C order, whatever the strides.
  let t1_contiguous = t1.is_C_Contiguous()
  let t2_contiguous = t2.is_C_Contiguous()

  # Get tensor data address with offset builtin
  var t1data = t1.dataArray
  var t2data = t2.dataArray

  # Optimize for loops in contiguous cases
  if t1_contiguous and t2_contiguous:
    for i in iter_offset..<(iter_offset+iter_size):
      dualStridedIterationYield(strider, t1data, t2data, i, i, i)
  elif t1_contiguous:
    initStridedIteration(t2_coord, t2_backstrides, t2_iter_pos, t2, iter_offset, iter_size)
    for i in iter_offset..<(iter_offset+iter_size):
      dualStridedIterationYield(strider, t1data, t2data, i, i, t2_iter_pos)
      advanceStridedIteration(t2_coord, t2_backstrides, t2_iter_pos, t2, iter_offset, iter_size)
  elif t2_contiguous:
    initStridedIteration(t1_coord, t1_backstrides, t1_iter_pos, t1, iter_offset, iter_size)
    for i in iter_offset..<(iter_offset+iter_size):
      dualStridedIterationYield(strider, t1data, t2data, i, t1_iter_pos, i)
      advanceStridedIteration(t1_coord, t1_backstrides, t1_iter_pos, t1, iter_offset, iter_size)
  else:
    initStridedIteration(t1_coord, t1_backstrides, t1_iter_pos, t1, iter_offset, iter_size)
    initStridedIteration(t2_coord, t2_backstrides, t2_iter_pos, t2, iter_offset, iter_size)
    for i in iter_offset..<(iter_offset+iter_size):
      dualStridedIterationYield(strider, t1data, t2data, i, t1_iter_pos, t2_iter_pos)
      advanceStridedIteration(t1_coord, t1_backstrides, t1_iter_pos, t1, iter_offset, iter_size)
      advanceStridedIteration(t2_coord, t2_backstrides, t2_iter_pos, t2, iter_offset, iter_size)

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
