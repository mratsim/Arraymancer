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

proc check_reshape(t: Tensor, new_shape:seq[int]) =
  if t.shape.product != new_shape.product:
    raise newException(ValueError, "The total number of elements in the old and the new reshaped matrix must be the same")


proc transpose*(t: Tensor): Tensor {.noSideEffect.}=
  ## Transpose a Tensor. For N-d Tensor with shape (0, 1, 2 ... n-1)
  ## the resulting tensor will have shape (n-1, ... 2, 1, 0)
  ## Data is copied as is and not modified.

  result.shape = t.shape.reversed
  result.strides = t.strides.reversed
  result.offset = t.offset
  result.data = t.data

proc asContiguous*[B,T](t: Tensor[B,T]): Tensor[B,T] {.noSideEffect.}=
  ## Transform a tensor with general striding to a Row major Tensor

  if t.isContiguous: return t

  result.shape = t.shape
  result.strides = shape_to_strides(result.shape)
  result.offset = 0
  result.data = newSeq[T](result.shape.product)

  var i = 0 ## TODO: use pairs/enumerate instead - pending https://forum.nim-lang.org/t/2972
  for val in t:
    result.data[i] = val
    inc i

proc reshape*[B,T](t: Tensor[B,T], new_shape: varargs[int]): Tensor[B,T] {.noSideEffect.}=
  ## Reshape a tensor
  ## TODO: tests
  ## TODO: fuse toTensor.reshape

  let ns = @new_shape
  when compileOption("boundChecks"): check_reshape(t, ns)

  # Can't call "tensor" template here for some reason
  result.shape = ns
  result.strides = shape_to_strides(result.shape)
  result.offset = 0
  result.data = newSeq[T](result.shape.product)

  var i = 0 ## TODO: use pairs/enumerate instead - pending https://forum.nim-lang.org/t/2972
  for val in t:
    result.data[i] = val
    inc i

proc broadcast*[B,T](t: Tensor[B,T], shape: openarray[int]): Tensor[B,T] {.noSideEffect.}=
  ## Broadcasting array
  ## Todo: proper bound-checking
  ## todo: testing
  ## TODO: use term-rewriting macro to have t1.bc * t2.bc broadcasted in compatible shape
  result = t
  assert t.rank == shape.len

  for i in 0..<result.rank:
    if result.shape[i] == 1:
      if shape[i] != 1:
        result.shape[i] = shape[i]
        result.strides[i] = 0
    elif result.shape[i] != shape[i]:
      raise newException(ValueError, "The broadcasted size of the tensor must match existing size for non-singleton dimension")

template bc*(t: Tensor, shape: openarray[int]): untyped =
  ## Alias
  t.broadcast(shape)

proc exch_dim(t: Tensor, dim1, dim2: int): Tensor {.noSideEffect.}=
  if dim1 == dim2:
    return t
  
  result = t
  swap(result.strides[dim1], result.strides[dim2])
  swap(result.shape[dim1], result.shape[dim2])

proc permute*(t: Tensor, dims: varargs[int]): Tensor {.noSideEffect.}=
  
  # TODO: bounds check
  var perm = @dims
  result = t
  for i, p in perm:
    if p != i and p != -1:
      var j = i
      while true:
        result = result.exch_dim(j, perm[j])
        (perm[j], j) = (-1, perm[j])
        if perm[j] == i:
          break
      perm[j] = -1