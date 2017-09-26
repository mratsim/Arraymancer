# Copyright 2017 Mamy André-Ratsimbazafy
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

proc check_index(t: Tensor, idx: varargs[int]) {.noSideEffect.}=
  if idx.len != t.rank:
    raise newException(IndexError, "Number of arguments: " &
                    $(idx.len) &
                    ", is different from tensor rank: " &
                    $(t.rank))

proc check_elementwise(a, b:AnyTensor)  {.noSideEffect.}=
  ## Check if element-wise operations can be applied to 2 Tensors
  if a.shape != b.shape:
    raise newException(ValueError, "Both Tensors should have the same shape")

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

type
  IterKind = enum
    Values, MemOffset, Coord_Values, MemOffset_Values

template strided_iteration[T](t: Tensor[T], strider: IterKind): untyped =
  ## Iterate over a Tensor, displaying data as in C order, whatever the strides.

  block:
    let size = t.shape.product

    when strider != IterKind.Coord_Values:
      if t.is_C_contiguous:
        for i in t.offset..<(t.offset+size):
          when strider == IterKind.Values: yield t.data[i]
          when strider == IterKind.MemOffset: yield i
          elif strider == IterKind.MemOffset_Values: yield (i, t.data[i])
        break

    ## Iterator init
    var iter_pos = t.offset
    let rank = t.rank
    var coord: array[MAXRANK, int]
    var backstrides {.noInit.}: array[MAXRANK, int]
    for i in 0..<rank:
      backstrides[i] = t.strides[i]*(t.shape[i]-1)

    ## Iterator loop
    for i in 0..<size:
      ## Templating the return value
      when strider == IterKind.Values: yield t.data[iter_pos]
      elif strider == IterKind.Coord_Values: yield (coord[0..<rank], t.data[iter_pos])
      elif strider == IterKind.MemOffset: yield iter_pos
      elif strider == IterKind.MemOffset_Values: yield (iter_pos, t.data[iter_pos])

      ## Computing the next position
      for k in countdown(rank - 1,0):
        if coord[k] < t.shape[k]-1:
          coord[k] += 1
          iter_pos += t.strides[k]
          break
        else:
          coord[k] = 0
          iter_pos -= backstrides[k]

iterator items*[T](t: Tensor[T]): T {.noSideEffect.}=
  ## Inline iterator on Tensor values
  ##
  ## The iterator will iterate in C order regardingless of the tensor properties (Fortran layout, non-contiguous, slice ...).
  ## So [0, 0, 0] then [0, 0, 1] then ... then [0, 1, 0] ...
  ##
  ## Usage:
  ##  .. code:: nim
  ##     for val in t: # items is implicitly called
  ##       val += 42
  t.strided_iteration(IterKind.Values)

proc values*[T](t: Tensor[T]): auto {.noSideEffect.}=
  ## Creates an closure iterator on Tensor values.
  ##
  ## The iterator will iterate in C order regardingless of the tensor properties (Fortran layout, non-contiguous, slice ...).
  ## So [0, 0, 0] then [0, 0, 1] then ... then [0, 1, 0] ...
  ##
  ## A closure iterator must be assigned to an iterator variable first.
  ## Usage:
  ##  .. code:: nim
  ##     let it = t.values
  ##     for val in it():
  ##       # do stuff
  ## Note: This is mostly useful for iterator chaining. Prefer the inline iterator ``items`` for simple iteration.
  ##
  ##  .. code:: nim
  ##     for val in t: # The `for` loop call the items iterator implicitly.
  ##       # do stuff
  ## Contrary to other ndarray packages looping in Arraymancer is not slow.
  let ref_t = t.unsafeAddr # avoid extra copy
  return iterator(): T = ref_t[].strided_iteration(IterKind.Values)

iterator mitems*[T](t: var Tensor[T]): var T {.noSideEffect.}=
  ## Inline iterator on Tensor values.
  ## Values yielded can be directly modified
  ## and avoid bound-checking/index calculation with t[index] = val.
  ##
  ## The iterator will iterate in C order regardingless of the tensor properties (Fortran layout, non-contiguous, slice ...).
  ## So [0, 0, 0] then [0, 0, 1] then ... then [0, 1, 0] ...
  ##
  ## Usage:
  ##  .. code:: nim
  ##     for val in t.mitems:
  ##       val += 42
  t.strided_iteration(IterKind.Values)

iterator noncontiguous_mitems*[T](t: var Tensor[T]): var T {.noSideEffect.}=
  ## Inline iterator on Tensor values.
  ## Values yielded can be directly modified
  ## The iterable order is non contiguous.
  if t.isFullyIterable():
    for i in 0..<t.data.len:
      yield t.data[i]
  else:
    t.strided_iteration(IterKind.Values)

iterator pairs*[T](t: Tensor[T]): (seq[int], T) {.noSideEffect.}=
  ## Inline iterator on Tensor (coordinates, values)
  ##
  ## The iterator will iterate in C order regardingless of the tensor properties (Fortran layout, non-contiguous, slice ...).
  ## So [0, 0, 0] then [0, 0, 1] then ... then [0, 1, 0] ...
  ##
  ## It returns a tuple of (coordinates, value) like (@[1,0,1], 1337)
  ##
  ## Usage:
  ##  .. code:: nim
  ##     for coord, val in t: # pairs is implicitly called
  ##       echo coord
  ##       echo val
  ##  .. code:: nim
  ##     for coordval in t.pairs: # pairs is explicitly called
  ##       echo coordval[0]
  ##       echo coordval[1]
  t.strided_iteration(IterKind.Coord_Values)

iterator real_indices(t: Tensor): int {.noSideEffect.}=
  ## Inline stride-aware iterator on Tensor real indices in the seq storage
  ## For loop will automatically use this one. (A closure iterator do not implement "items")
  t.strided_iteration(IterKind.MemOffset)

proc real_indices(t: Tensor): auto {.noSideEffect.}=
  ## Closure stride-aware iterator on Tensor real indices in the seq storage
  ## For loop will not use this one. It must be assigned before use.
  let ref_t = t.unsafeAddr # avoid extra copy
  return iterator(): int = ref_t[].strided_iteration(IterKind.MemOffset)

template axis_iterator[T](t: Tensor[T], axis: int): untyped =
  var out_t = t

  let axis_len = t.shape[axis]
  let axis_stride = t.strides[axis]
  out_t.shape[axis] = 1

  for _ in 0..<axis_len:
    yield out_t
    out_t.offset += axis_stride

iterator axis*[T](t: Tensor[T], axis: int): Tensor[T] {.noSideEffect.}=
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


iterator zip*[T1, T2](a: Tensor[T1], b: Tensor[T2]): tuple[a: T1, b: T2] =
  ## Iterates simultaneously on two tensors returning their elements in a tuple.
  ## As a shortcut elements from the tuple can be addressed via result.a and result.b
  ## Note: only tensors of the same shape will be zipped together.
  when compileOption("boundChecks"):
    check_elementwise(a,b)

  for vals in zip(a.values, b.values): # TODO: use inline iterators
    yield vals