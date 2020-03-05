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

import  ../backend/[global_config, memory_optimization_hints],
        ../data_structure,
        ./p_checks

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
#   for i in 0 ..< t.shape.product:
#
#     ## Templating the return value
#     when strider == IterKind.Values: yield t.unsafe_raw_buf[iter_pos]
#     elif strider == IterKind.Coord_Values: yield (coord, t.unsafe_raw_buf[iter_pos])
#     elif strider == IterKind.MemOffset: yield iter_pos
#     elif strider == IterKind.MemOffset_Values: yield (iter_pos, t.unsafe_raw_buf[iter_pos])
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

proc getIndex*[T](t: Tensor[T], idx: varargs[int]): int {.noSideEffect,inline.} =
  ## Convert [i, j, k, l ...] to the proper index.
  when compileOption("boundChecks"):
    t.check_index(idx)

  result = t.offset
  for i in 0..<idx.len:
    result += t.strides[i]*idx[i]

proc getContiguousIndex*[T](t: Tensor[T], idx: int): int {.noSideEffect,inline.} =
  result = t.offset
  if idx != 0:
    var z = 1
    for i in countdown(t.rank - 1,0):
      let coord = (idx div z) mod t.shape[i]
      result += coord*t.strides[i]
      z *= t.shape[i]

proc atIndex*[T](t: Tensor[T], idx: varargs[int]): T {.noSideEffect,inline.} =
  ## Get the value at input coordinates
  ## This used to be `[]` before slicing was implemented
  result = t.unsafe_raw_buf[t.getIndex(idx)]

proc atIndex*[T](t: var Tensor[T], idx: varargs[int]): var T {.noSideEffect,inline.} =
  ## Get the value at input coordinates
  ## This allows inplace operators t[1,2] += 10 syntax
  result = t.unsafe_raw_buf[t.getIndex(idx)]

proc atIndexMut*[T](t: var Tensor[T], idx: varargs[int], val: T) {.noSideEffect,inline.} =
  ## Set the value at input coordinates
  ## This used to be `[]=` before slicing was implemented
  t.unsafe_raw_buf[t.getIndex(idx)] = val

## Iterators
type
  IterKind* = enum
    Values, Iter_Values, Offset_Values

template initStridedIteration*(coord, backstrides, iter_pos: untyped, t, iter_offset, iter_size: typed): untyped =
  ## Iterator init
  var iter_pos = 0
  withMemoryOptimHints() # MAXRANK = 8, 8 ints = 64 Bytes, cache line = 64 Bytes --> profit !
  var coord {.align64, noInit.}: array[MAXRANK, int]
  var backstrides {.align64, noInit.}: array[MAXRANK, int]
  for i in 0..<t.rank:
    backstrides[i] = t.strides[i]*(t.shape[i]-1)
    coord[i] = 0

  # Calculate initial coords and iter_pos from iteration offset
  if iter_offset != 0:
    var z = 1
    for i in countdown(t.rank - 1,0):
      coord[i] = (iter_offset div z) mod t.shape[i]
      iter_pos += coord[i]*t.strides[i]
      z *= t.shape[i]

template advanceStridedIteration*(coord, backstrides, iter_pos, t, iter_offset, iter_size: typed): untyped =
  ## Computing the next position
  for k in countdown(t.rank - 1,0):
    if coord[k] < t.shape[k]-1:
      coord[k] += 1
      iter_pos += t.strides[k]
      break
    else:
      coord[k] = 0
      iter_pos -= backstrides[k]

template stridedIterationYield*(strider: IterKind, data, i, iter_pos: typed) =
  ## Iterator the return value
  when strider == IterKind.Values: yield data[iter_pos]
  elif strider == IterKind.Iter_Values: yield (i, data[iter_pos])
  elif strider == IterKind.Offset_Values: yield (iter_pos, data[iter_pos]) ## TODO: remove workaround for C++ backend

template stridedIteration*(strider: IterKind, t, iter_offset, iter_size: typed): untyped =
  ## Iterate over a Tensor, displaying data as in C order, whatever the strides.

  # Get tensor data address with offset builtin
  let data = t.unsafe_raw_offset()

  # Optimize for loops in contiguous cases
  if t.is_C_Contiguous:
    for i in iter_offset..<(iter_offset+iter_size):
      stridedIterationYield(strider, data, i, i)
  else:
    initStridedIteration(coord, backstrides, iter_pos, t, iter_offset, iter_size)
    for i in iter_offset..<(iter_offset+iter_size):
      stridedIterationYield(strider, data, i, iter_pos)
      advanceStridedIteration(coord, backstrides, iter_pos, t, iter_offset, iter_size)

template stridedCoordsIteration*(t, iter_offset, iter_size: typed): untyped =
  ## Iterate over a Tensor, displaying data as in C order, whatever the strides. (coords)

  # Get tensor data address with offset builtin
  let data = t.unsafe_raw_offset()
  let rank = t.rank

  initStridedIteration(coord, backstrides, iter_pos, t, iter_offset, iter_size)
  for i in iter_offset..<(iter_offset+iter_size):
    yield (coord[0..<rank], data[iter_pos])
    advanceStridedIteration(coord, backstrides, iter_pos, t, iter_offset, iter_size)

template dualStridedIterationYield*(strider: IterKind, t1data, t2data, i, t1_iter_pos, t2_iter_pos: typed) =
  ## Iterator the return value
  when strider == IterKind.Values: yield (t1data[t1_iter_pos], t2data[t2_iter_pos])
  elif strider == IterKind.Iter_Values: yield (i, t1data[t1_iter_pos], t2data[t2_iter_pos])
  elif strider == IterKind.Offset_Values: yield (t1_iter_pos, t1data[t1_iter_pos], t2data[t2_iter_pos])  ## TODO: remove workaround for C++ backend


template dualStridedIteration*(strider: IterKind, t1, t2, iter_offset, iter_size: typed): untyped =
  ## Iterate over two Tensors, displaying data as in C order, whatever the strides.
  let t1_contiguous = t1.is_C_Contiguous()
  let t2_contiguous = t2.is_C_Contiguous()

  # Get tensor data address with offset builtin
  let t1data = t1.unsafe_raw_offset()
  let t2data = t2.unsafe_raw_offset()

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

template tripleStridedIterationYield*(strider: IterKind, t1data, t2data, t3data, i, t1_iter_pos, t2_iter_pos, t3_iter_pos: typed) =
  ## Iterator the return value
  when strider == IterKind.Values: yield (t1data[t1_iter_pos], t2data[t2_iter_pos], t3data[t3_iter_pos])
  elif strider == IterKind.Iter_Values: yield (i, t1data[t1_iter_pos], t2data[t2_iter_pos], t3data[t3_iter_pos])
  elif strider == IterKind.Offset_Values: yield (t1_iter_pos, t1data[t1_iter_pos], t2data[t2_iter_pos], t3data[t3_iter_pos])  ## TODO: remove workaround for C++ backend

template tripleStridedIteration*(strider: IterKind, t1, t2, t3, iter_offset, iter_size: typed): untyped =
  ## Iterate over two Tensors, displaying data as in C order, whatever the strides.
  let t1_contiguous = t1.is_C_Contiguous()
  let t2_contiguous = t2.is_C_Contiguous()
  let t3_contiguous = t3.is_C_Contiguous()

  # Get tensor data address with offset builtin
  withMemoryOptimHints()
  let t1data = t1.unsafe_raw_offset()
  let t2data = t2.unsafe_raw_offset()
  let t3data = t3.unsafe_raw_offset()

  # Optimize for loops in contiguous cases
  # Note that not all cases are handled here, just some probable ones
  if t1_contiguous and t2_contiguous and t3_contiguous:
    for i in iter_offset..<(iter_offset+iter_size):
      tripleStridedIterationYield(strider, t1data, t2data, t3data, i, i, i, i)
  elif t1_contiguous and t2_contiguous:
    initStridedIteration(t3_coord, t3_backstrides, t3_iter_pos, t3, iter_offset, iter_size)
    for i in iter_offset..<(iter_offset+iter_size):
      tripleStridedIterationYield(strider, t1data, t2data, t3data, i, i, i, t3_iter_pos)
      advanceStridedIteration(t3_coord, t3_backstrides, t3_iter_pos, t3, iter_offset, iter_size)
  elif t1_contiguous:
    initStridedIteration(t2_coord, t2_backstrides, t2_iter_pos, t2, iter_offset, iter_size)
    initStridedIteration(t3_coord, t3_backstrides, t3_iter_pos, t3, iter_offset, iter_size)
    for i in iter_offset..<(iter_offset+iter_size):
      tripleStridedIterationYield(strider, t1data, t2data, t3data, i, i, t2_iter_pos, t3_iter_pos)
      advanceStridedIteration(t2_coord, t2_backstrides, t2_iter_pos, t2, iter_offset, iter_size)
      advanceStridedIteration(t3_coord, t3_backstrides, t3_iter_pos, t3, iter_offset, iter_size)
  else:
    initStridedIteration(t1_coord, t1_backstrides, t1_iter_pos, t1, iter_offset, iter_size)
    initStridedIteration(t2_coord, t2_backstrides, t2_iter_pos, t2, iter_offset, iter_size)
    initStridedIteration(t3_coord, t3_backstrides, t3_iter_pos, t3, iter_offset, iter_size)
    for i in iter_offset..<(iter_offset+iter_size):
      tripleStridedIterationYield(strider, t1data, t2data, t3data, i, t1_iter_pos, t2_iter_pos, t3_iter_pos)
      advanceStridedIteration(t1_coord, t1_backstrides, t1_iter_pos, t1, iter_offset, iter_size)
      advanceStridedIteration(t2_coord, t2_backstrides, t2_iter_pos, t2, iter_offset, iter_size)
      advanceStridedIteration(t3_coord, t3_backstrides, t3_iter_pos, t3, iter_offset, iter_size)
