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

proc getIndex[B: static[Backend], T](t: Tensor[B,T], idx: varargs[int]): int {.noSideEffect.} =
  ## Convert [i, j, k, l ...] to the proper index.
  when compileOption("boundChecks"):
    t.check_index(idx)

  var real_idx = t.offset
  for i,j in zip(t.strides,idx):
    real_idx += i*j
  return real_idx

proc atIndex*[B: static[Backend], T](t: Tensor[B,T], idx: varargs[int]): T {.noSideEffect.} =
  ## Get the value at input coordinates
  ## This used to be `[]` before slicing was implemented
  return t.data[t.getIndex(idx)]

proc atIndexMut*[B: static[Backend], T](t: var Tensor[B,T], idx: varargs[int], val: T) {.noSideEffect.} =
  ## Set the value at input coordinates
  ## This used to be `[]=` before slicing was implemented
  t.data[t.getIndex(idx)] = val

type
  IterKind = enum
    Values, MemOffset, Coord_Values, MemOffset_Values

template strided_iteration[B,T](t: Tensor[B,T], strider: IterKind): untyped =
  ## Iterate over a Tensor, displaying data as in C order, whatever the strides.

  ## Iterator init
  var coord = newSeq[int](t.rank) # Coordinates in the n-dimentional space
  var backstrides: seq[int] = @[] # Offset between end of dimension and beginning
  for i,j in zip(t.strides,t.shape): backstrides.add(i*(j-1))

  var iter_pos = t.offset

  ## Iterator loop
  for i in 0 .. <t.shape.product:

    ## Templating the return value
    when strider == IterKind.Values: yield t.data[iter_pos]
    elif strider == IterKind.Coord_Values: yield (coord, t.data[iter_pos])
    elif strider == IterKind.MemOffset: yield iter_pos
    elif strider == IterKind.MemOffset_Values: yield (iter_pos, t.data[iter_pos])

    ## Computing the next position
    for k in countdown(t.rank - 1,0):
      if coord[k] < t.shape[k]-1:
        coord[k] += 1
        iter_pos += t.strides[k]
        break
      else:
        coord[k] = 0
        iter_pos -= backstrides[k]

iterator items*[B,T](t: Tensor[B,T]): T {.noSideEffect.}=
  ## Inline stride-aware iterator on Tensor values
  t.strided_iteration(IterKind.Values)

proc values*[B,T](t: Tensor[B,T]): auto {.noSideEffect.}=
  ## Closure stride-aware iterator on Tensor values
  return iterator(): T = t.strided_iteration(IterKind.Values)

iterator pairs*[B,T](t: Tensor[B,T]): (seq[int], T) {.noSideEffect.}=
  ## Inline stride-aware iterator on Tensor coordinates i.e. [1,2,3] and values
  t.strided_iteration(IterKind.Coord_Values)

iterator real_indices(t: Tensor): int {.noSideEffect.}=
  ## Inline stride-aware iterator on Tensor real indices in the seq storage
  ## For loop will automatically use this one. (A closure iterator do not implement "items")
  t.strided_iteration(IterKind.MemOffset)

proc real_indices(t: Tensor): auto {.noSideEffect.}=
  ## Closure stride-aware iterator on Tensor real indices in the seq storage
  ## For loop will not use this one. It must be assigned before use.
  return iterator(): int = t.strided_iteration(IterKind.MemOffset)

iterator axis*[B,T](t: Tensor[B,T], axis: int): Tensor[B,T] {.noSideEffect.}=
  ## Iterates over an axis

  var out_t = t
  let axis_len = t.shape[axis]
  let axis_stride = t.strides[axis]
  out_t.shape[axis] = 1

  for _ in 0..<axis_len:
    yield out_t
    out_t.offset += axis_stride