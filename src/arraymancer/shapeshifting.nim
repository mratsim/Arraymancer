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

proc check_concat(t1, t2: Tensor, axis: int) {.noSideEffect,inline.}=
  let check1 = t1.shape[0..<axis] == t2.shape[0..<axis]
  let check2 = t2.shape[axis+1..t1.shape.high] == t2.shape[axis+1..t2.shape.high]

  if not check1 or not check2:
    raise newException(ValueError, "Concatenation Error: Except along the concatenation axis tensors must have the same shape")

proc transpose*(t: Tensor): Tensor {.noSideEffect.}=
  ## Transpose a Tensor.
  ##
  ## For N-d Tensor with shape (0, 1, 2 ... n-1) the resulting tensor will have shape (n-1, ... 2, 1, 0)
  ##
  ## Data is copied as-is and not modified.

  result.shape = t.shape.reversed
  result.strides = t.strides.reversed
  result.offset = t.offset
  result.data = t.data

proc asContiguous*[T](t: Tensor[T], layout: OrderType = rowMajor, force: bool = false): Tensor[T] {.noSideEffect.}=
  ## Transform a tensor with general striding to a Tensor with contiguous layout.
  ## By default tensor will be rowMajor.
  ## By default nothing is done if the tensor is already contiguous (C Major or F major)
  ## The "force" parameter can force re-ordering to a specific layout

  if t.isContiguous and not force:
    return t
  elif t.is_C_contiguous and layout == rowMajor:
    return t
  elif t.is_F_contiguous and layout == colMajor:
    return t

  result.shape = t.shape
  result.strides = shape_to_strides(result.shape, layout)
  result.offset = 0
  result.data = newSeq[T](result.shape.product)

  var i = 0 ## TODO: use pairs/enumerate instead - pending https://forum.nim-lang.org/t/2972

  if layout == rowMajor:
    for val in t:
      result.data[i] = val
      inc i
  else:
    for val in t.transpose:
      result.data[i] = val
      inc i

proc reshape_with_copy[T](t: Tensor[T], new_shape: seq[int]): Tensor[T] {.noSideEffect.}=
  # Can't call "tensorCpu" template here for some reason
  result.shape = new_shape
  result.strides = shape_to_strides(result.shape)
  result.offset = 0
  result.data = newSeq[T](result.shape.product)

  var i = 0 ## TODO: use pairs/enumerate instead - pending https://forum.nim-lang.org/t/2972
  for val in t:
    result.data[i] = val
    inc i

# The following needs benchmark, it seems slower by 120ms on my machine (Broadwell i7-5XXX U mobile)
# than reshape with copy when reshaping 25000 elements to a 5000x5000 matrices

# proc reshape_no_copy[B,T](t: Tensor[B,T], new_shape: seq[int]): Tensor[B,T] {.noSideEffect.}=
#   ## For contiguous arrays
#   var matched_dims = 0
#   for shapes in zip(t.shape, new_shape):
#     if shapes[0] != shapes[1]:
#       break
#     inc matched_dims
#
#   result.shape = new_shape
#
#   # Strides extended for unmatched dimension
#   let ext_strides = result.shape[matched_dims..result.shape.high].shape_to_strides
#
#   result.strides = t.strides[0..<matched_dims] & ext_strides
#   result.offset = 0
#   result.data = t.data

proc reshape*(t: Tensor, new_shape: varargs[int]): Tensor {.noSideEffect.}=
  ## Reshape a tensor
  ## Input:
  ##   - a tensor
  ##   - a new shape. Number of elements must be the same
  ## Returns:
  ##   - a tensor with the same data but reshaped.

  let ns = @new_shape
  when compileOption("boundChecks"): check_reshape(t, ns)
  
  #if t.isContiguous:
  #  return t.reshape_no_copy(ns)
  return t.reshape_with_copy(ns)

proc broadcast*[T](t: Tensor[T], shape: openarray[int]): Tensor[T] {.noSideEffect.}=
  ## Broadcast array
  ##
  ## Dimension(s) of size 1 can be expanded to arbitrary size by replicating
  ## values along that dimension.
  # Todo: proper bound-checking
  # todo: testing
  # TODO: use term-rewriting macro to have t1.bc * t2.bc broadcasted in compatible shape
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
  ## Alias for ``broadcast``
  t.broadcast(shape)

proc exch_dim(t: Tensor, dim1, dim2: int): Tensor {.noSideEffect.}=
  if dim1 == dim2:
    return

  result = t
  swap(result.strides[dim1], result.strides[dim2])
  swap(result.shape[dim1], result.shape[dim2])

proc permute*(t: Tensor, dims: varargs[int]): Tensor {.noSideEffect.}=
  ## Permute dimensions
  ## Input:
  ##   - a tensor
  ##   - the new dimension order
  ## Returns:
  ##   - a tensor with re-order dimension
  ## Example:
  ##
  ## a.permute(0,2,1) # dim 0 stays at 0, dim 1 becomes dim 2 and dim 2 becomes dim 1

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


proc concat*[T](t_list: varargs[Tensor[T]], axis: int): Tensor[T]  {.noSideEffect.}=
  ## Concatenate tensors
  ## Input:
  ##   - Tensors
  ##   - An axis (dimension)
  ## Returns:
  ##   - a tensor
  var axis_dim = 0
  let t0 = t_list[0]

  for t in t_list:
    when compileOption("boundChecks"):
      check_concat(t0, t, axis)
    axis_dim += t.shape[axis]

  let concat_shape = t0.shape[0..<axis] & axis_dim & t0.shape[axis+1..t0.shape.high]

  ## Setup the Tensor
  result.shape = concat_shape
  result.strides = shape_to_strides(result.shape)
  result.offset = 0
  result.data = newSeq[T](result.shape.product)

  # Fill in the copy with the matching values
  var slices = concat_shape.mapIt((0..<it)|1)
  var iaxis = 0

  for t in t_list:
    slices[axis].a = iaxis
    slices[axis].b = iaxis + t.shape[axis] - 1
    result.slicerMut(slices, t)
    iaxis += t.shape[axis]