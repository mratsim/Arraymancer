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
        ../../private/ast_utils,
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

type TensorForm = object
  shape: Metadata
  strides: Metadata

proc rank(t: TensorForm): range[0 .. LASER_MAXRANK] {.inline.} =
  t.shape.len

func size(t: TensorForm): int {.inline.} =
  result = 1
  for i in 0..<t.rank:
    result *= t.shape[i]

func reduceRank(t: TensorForm): TensorForm =
  result = t

  var i = 0
  result.shape[0] = t.shape[0]
  result.strides[0] = t.strides[0]
  for j in 1..<t.rank:
    #spurious axis
    if t.shape[j] == 1:
      continue

    #current axis is spurious
    if result.shape[i] == 1:
      result.shape[i] = t.shape[j]
      result.strides[i] = t.strides[j]
      continue

    #axes can be coalesced
    if result.strides[i] == t.shape[j]*t.strides[j]:
      result.shape[i] = result.shape[i]*t.shape[j]
      result.strides[i] = t.strides[j]
      continue

    i += 1
    result.shape[i] = t.shape[j]
    result.strides[i] = t.strides[j]
  result.shape.len = i + 1
  result.strides.len = i + 1

func floor(x: int, divisor: int): int {.inline.} =
  return divisor*(x div divisor)

func ceil(x: int, divisor: int): int {.inline.} =
  return divisor*(((x - 1) div divisor) + 1)

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
  when T is KnownSupportsCopyMem:
    result = t.unsafe_raw_buf[t.getIndex(idx)]
  else:
    result = t.storage.raw_buffer[t.getIndex(idx)]

proc atIndex*[T](t: var Tensor[T], idx: varargs[int]): var T {.noSideEffect,inline.} =
  ## Get the value at input coordinates
  ## This allows inplace operators t[1,2] += 10 syntax
  when T is KnownSupportsCopyMem:
    result = t.unsafe_raw_buf[t.getIndex(idx)]
  else:
    result = t.storage.raw_buffer[t.getIndex(idx)]

proc atIndexMut*[T](t: var Tensor[T], idx: varargs[int], val: T) {.noSideEffect,inline.} =
  ## Set the value at input coordinates
  ## This used to be `[]=` before slicing was implemented
  when T is KnownSupportsCopyMem:
    t.unsafe_raw_buf[t.getIndex(idx)] = val
  else:
    t.storage.raw_buffer[t.getIndex(idx)] = val

#[
The following accessors represent a ``very`` specific workaround.
The templates used for the iterators in this file make use of `unsafe_raw_offset`.
This is not valid for `not KnownSupportsCopyMem` types. That's why we
define these helper accessors, which access the corresponding position
for `seq` based Tensors, including the offset!
Instead of defining a `Raw(Im)MutableView` type, we simply define a template
to the input tensor, like so:

 .. code-block:: nim
   when getSubType(t) is KnownSupportsCopyMem:
     let data = t.unsafe_raw_offset()
   else:
     template data: untyped = t

The `data` template is then given to the following code, which simply accesses
the input tensor. Since it is a seq based tensor, it will use the accessors
below.
]#

func `[]`[T: not KnownSupportsCopyMem](t: Tensor[T], idx: int): T =
  t.storage.raw_buffer[t.offset + idx]
func `[]`[T: not KnownSupportsCopyMem](t: var Tensor[T], idx: int): var T =
  t.storage.raw_buffer[t.offset + idx]
func `[]=`[T: not KnownSupportsCopyMem](t: var Tensor[T], idx: int, val: T) =
    t.storage.raw_buffer[t.offset + idx] = val

## Iterators
type
  IterKind* = enum
    Values, Iter_Values, Offset_Values

template initStridedIteration*(coord, backstrides, iter_pos: untyped, t, iter_offset, iter_size: typed): untyped =
  ## Iterator init
  var iter_pos = 0
  withMemoryOptimHints() # MAXRANK = 8, 8 ints = 64 Bytes, cache line = 64 Bytes --> profit !
  var coord {.align64, noinit.}: array[MAXRANK, int]
  var backstrides {.align64, noinit.}: array[MAXRANK, int]
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

template stridedIterationLoop*(strider: IterKind, data, t, iter_offset, iter_size, prev_d, last_d: typed) =
  ## We break up the tensor in 5 parts and iterate over each using for loops.
  ## We do this because the loop ranges and nestedness are different for each part.
  ## The part boundaries are calculated and stored in the `bp1`, `bp2`, `bp3`
  ## and `bp4` variables. The `(iter_offset, bp1)` segment is a rank-1 tensor
  ## of size `<last_d`. The `(bp1, bp2)` segment is a rank-2 tensor with first
  ## axis smaller than `prev_d`. The `(bp2, bp3)` segment is the main body, an
  ## rank-n tensor with last axes sizes `prev_d` and `last_d`. The `(bp3, bp4)`
  ## segment is a rank-2 tensor, and the `(bp4, iter_offset + iter_size)` segment
  ## is a rank-1 tensor.
  assert t.rank > 1

  let prev_s = t.strides[^2]
  let last_s = t.strides[^1]
  let rank = t.rank
  let size = t.size

  assert iter_offset >= 0
  assert iter_size <= size - iter_offset
  assert prev_d > 0 and last_d > 0
  assert size mod prev_d*last_d == 0

  initStridedIteration(coord, backstrides, iter_pos, t, iter_offset, iter_size)

  let bp1 =
    if iter_offset == 0:
      0
    else:
      min(iter_offset + iter_size, ceil(iter_offset, last_d))
  let bp2 =
    if iter_offset == 0:
      0
    else:
      max(bp1, min(floor(iter_offset + iter_size, prev_d*last_d), ceil(iter_offset, prev_d*last_d)))
  let bp3 =
    if iter_size == size:
      size
    else:
      max(bp2, floor(iter_offset + iter_size, prev_d*last_d))
  let bp4 =
    if iter_size == size:
      size
    else:
      max(bp3, floor(iter_offset + iter_size, last_d))

  assert iter_offset <= bp1 and bp1 <= bp2 and bp2 <= bp3 and bp3 <= bp4 and bp4 <= iter_offset + iter_size
  assert bp1 - iter_offset < last_d and (bp1 mod last_d == 0 or bp1 == iter_offset + iter_size)
  assert bp2 == bp1 or (bp2 mod prev_d*last_d == 0 and bp2 - bp1 < prev_d*last_d)
  assert bp3 == bp2 or bp3 mod prev_d*last_d == 0
  assert bp4 == bp3 or (bp4 mod last_d == 0 and bp4 - bp3 < prev_d*last_d)
  assert iter_offset + iter_size - bp4 < last_d

  var i = iter_offset

  if bp1 > iter_offset:
    coord[rank - 1] += bp1 - i - 1
    while i < bp1:
      stridedIterationYield(strider, data, i, iter_pos)
      iter_pos += last_s
      i += 1
    iter_pos -= last_s
    advanceStridedIteration(coord, backstrides, iter_pos, t, iter_offset, iter_size)

  if bp2 > bp1:
    coord[rank - 2] += ((bp2 - i) div last_d) - 1
    coord[rank - 1] = last_d - 1
    while i < bp2:
      for _ in 0..<last_d:
        stridedIterationYield(strider, data, i, iter_pos)
        iter_pos += last_s
        i += 1
      iter_pos += prev_s - last_s*last_d
    iter_pos += last_s*(last_d - 1) - prev_s
    advanceStridedIteration(coord, backstrides, iter_pos, t, iter_offset, iter_size)

  while i < bp3:
    for _ in 0..<prev_d:
      for _ in 0..<last_d:
        stridedIterationYield(strider, data, i, iter_pos)
        iter_pos += last_s
        i += 1
      iter_pos += prev_s - last_s*last_d
    iter_pos -= prev_s*prev_d

    for k in countdown(rank - 3, 0):
      if coord[k] < t.shape[k] - 1:
        coord[k] += 1
        iter_pos += t.strides[k]
        break
      else:
        coord[k] = 0
        iter_pos -= backstrides[k]

  if bp4 > bp3:
    coord[rank - 2] += ((bp4 - i) div last_d) - 1
    coord[rank - 1] = last_d - 1
    while i < bp4:
      for _ in 0..<last_d:
        stridedIterationYield(strider, data, i, iter_pos)
        iter_pos += last_s
        i += 1
      iter_pos += prev_s - last_s*last_d
    iter_pos += last_s*(last_d - 1) - prev_s
    advanceStridedIteration(coord, backstrides, iter_pos, t, iter_offset, iter_size)

  while i < iter_offset + iter_size:
    stridedIterationYield(strider, data, i, iter_pos)
    iter_pos += last_s
    i += 1

template stridedIteration*(strider: IterKind, t, iter_offset, iter_size: typed): untyped =
  ## Iterate over a Tensor, displaying data as in C order, whatever the strides.

  # Get tensor data address with offset builtin
  # only reading here, pointer access is safe even for ref types

  when getSubType(type(t)) is KnownSupportsCopyMem:
    let data = t.unsafe_raw_offset()
  else:
    template data: untyped {.gensym.} = t

  let tf = reduceRank(TensorForm(shape: t.shape, strides: t.strides))

  assert tf.rank >= 1
  if tf.rank == 1:
    let s = tf.strides[^1]
    for i in iter_offset..<(iter_offset+iter_size):
      stridedIterationYield(strider, data, i, i*s)
  else:
    let prev_d = tf.shape[^2]
    let last_d = tf.shape[^1]
    if prev_d == 2 and last_d == 2:
      stridedIterationLoop(strider, data, tf, iter_offset, iter_size, 2, 2)
    elif last_d == 2:
      stridedIterationLoop(strider, data, tf, iter_offset, iter_size, prev_d, 2)
    elif last_d == 3:
      stridedIterationLoop(strider, data, tf, iter_offset, iter_size, prev_d, 3)
    else:
      stridedIterationLoop(strider, data, tf, iter_offset, iter_size, prev_d, last_d)

template stridedCoordsIteration*(t, iter_offset, iter_size: typed): untyped =
  ## Iterate over a Tensor, displaying data as in C order, whatever the strides. (coords)

  # Get tensor data address with offset builtin
  # only reading here, pointer access is safe even for ref types
  when getSubType(type(t)) is KnownSupportsCopyMem:
    let data = t.unsafe_raw_offset()
  else:
    template data: untyped {.gensym.} = t
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

  let t1_contiguous = t1.is_C_contiguous()
  let t2_contiguous = t2.is_C_contiguous()

  when getSubType(type(t1)) is KnownSupportsCopyMem:
    let t1data = t1.unsafe_raw_offset()
  else:
    template t1data: untyped {.gensym.} = t1
  when getSubType(type(t2)) is KnownSupportsCopyMem:
    let t2data = t2.unsafe_raw_offset()
  else:
    template t2data: untyped {.gensym.} = t2

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
  let t1_contiguous = t1.is_C_contiguous()
  let t2_contiguous = t2.is_C_contiguous()
  let t3_contiguous = t3.is_C_contiguous()

  # Get tensor data address with offset builtin
  withMemoryOptimHints()
  when getSubType(type(t1)) is KnownSupportsCopyMem:
    let t1data = t1.unsafe_raw_offset()
  else:
    template t1data: untyped {.gensym.} = t1
  when getSubType(type(t2)) is KnownSupportsCopyMem:
    let t2data = t2.unsafe_raw_offset()
  else:
    template t2data: untyped {.gensym.} = t2
  when getSubType(type(t3)) is KnownSupportsCopyMem:
    let t3data = t3.unsafe_raw_offset()
  else:
    template t3data: untyped {.gensym.} = t3

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

import std / macros
import ../accessors_macros_syntax
proc checkValidSliceType*(n: NimNode)
proc validObjectType*(n: NimNode): bool =
  ## Checks if the given node `n` corresponds to an object type
  ## that is allowed as an argument to
  doAssert n.typeKind in {ntyObject, ntyGenericInst}
  let typ = n.getTypeInst
  # Hardcode the `hasType` calls, because for `bindSym` need a
  # static string. Unnecessary to produce calls via a macro from an array
  if hasType(typ, "SteppedSlice"): return true
  if hasType(typ, "Ellipsis"):     return true
  # NOTE: On Nim 2.0 if the argument is a `Tensor[T]` we only get a `nnkSym`
  # from which we cannot get the inner type with the `getType*` logic. That
  # means `hasType` below matches regardless of the generic argument. There's
  # not much we can do about that here.
  if hasType(typ, "Tensor"):       return true
  # On Nim > 2.0 we get a `nnkBracketExpr` and can indeed compare the inner
  # type as below.
  # construct the types we want to compare with
  let validTensorTypes = [getTypeInst(Tensor[int])[1],
                          getTypeInst(Tensor[bool])[1]]
  for t in validTensorTypes:
    if sameType(typ, t): return true

proc checkValidSliceType*(n: NimNode) =
  ## Checks if the given node `n` has a type, which is valid as an argument to
  ## the `[]` and `[]=` macros. It will raise a CT error in case it is not.
  ##
  ## TODO: Do we / should we allow other integer types than `tyInt` / `int`?
  const validTypes = {ntyInt, ntyObject, ntyArray, ntySequence, ntyGenericInst}
  # `ntyObject` requires to be `Span`, ...
  template raiseError(arg: untyped): untyped =
    let typ = arg.getTypeInst
    error("Invalid argument to `[]` / `[]=` accessor. Must be an integer, array[int/bool], " &
      "seq[int/bool], Tensor[int/bool] or slice, but found " & $arg.repr & " of type: `" &
      $typ.repr & "`.")
  for arg in n:
    case arg.typeKind
    of validTypes:
      if arg.typeKind in {ntyObject, ntyGenericInst} and not validObjectType(arg):
        raiseError(arg)
      elif arg.typeKind in {ntyArray, ntySequence}:
        # Need to check inner type!
        checkValidSliceType(arg.getTypeInst()[^1])
        break
      else:
        continue
    else:
      raiseError(arg)
