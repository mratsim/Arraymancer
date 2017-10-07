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

proc check_reshape(t: AnyTensor, new_shape:MetadataArray) {.noSideEffect, inline.}=
  if t.size != new_shape.product:
    raise newException(ValueError, "The total number of elements in the old (" &
                                    $t.size &
                                    ") and the new (" &
                                    $new_shape.product &
                                    ") reshaped tensor must be the same")

proc check_nocopyReshape(t: AnyTensor) {.noSideEffect, inline.}=
  if not t.isContiguous:
    raise newException(ValueError, "The tensor must be contiguous for reshape without copy")

proc check_concat(t1, t2: Tensor, axis: int) {.noSideEffect,inline.}=
  let check1 = t1.shape[0..<axis] == t2.shape[0..<axis]
  let check2 = t2.shape[axis+1..t1.shape.high] == t2.shape[axis+1..t2.shape.high]

  if not check1 or not check2:
    raise newException(ValueError, "Concatenation Error: Except along the concatenation axis tensors must have the same shape")

proc check_squeezeAxis(t: AnyTensor, axis: int) {.noSideEffect, inline.}=
  if axis >= t.rank:
    raise newException(ValueError, "The axis is out of range, axis is " & $axis & " while the tensor rank is " & $t.rank )

proc check_unsqueezeAxis(t: AnyTensor, axis: int) {.noSideEffect, inline.}=
  if t.rank == 0 or axis > t.rank or axis < 0:
    raise newException(ValueError, "The new axis is out of range, axis is " & $axis & " while the tensor rank is " & $t.rank )

proc transpose*(t: Tensor): Tensor {.noSideEffect, inline.}=
  ## Transpose a Tensor.
  ##
  ## For N-d Tensor with shape (0, 1, 2 ... n-1) the resulting tensor will have shape (n-1, ... 2, 1, 0)
  ##
  ## Data is copied as-is and not modified.
  result.shape = t.shape.reversed
  result.strides = t.strides.reversed
  result.offset = t.offset
  result.data = t.data


proc unsafeTranspose*(t: Tensor): Tensor {.noSideEffect, inline.}=
  ## Transpose a Tensor without copy.
  ##
  ## Warning ⚠:
  ##   This is a no-copy operation, data is shared with the input.
  ##   This proc does not guarantee that a ``let`` value is immutable.
  ##
  ## For N-d Tensor with shape (0, 1, 2 ... n-1) the resulting tensor will have shape (n-1, ... 2, 1, 0)
  result.shape = t.shape.reversed
  result.strides = t.strides.reversed
  result.offset = t.offset
  shallowCopy(result.data, t.data)

template contiguousT[T](result, t: Tensor[T], layout: OrderType): untyped=
  if layout == rowMajor:
    result = t.mapT(x)
  else:
    result = t.transpose().mapT(x)

proc asContiguous*[T](t: Tensor[T], layout: OrderType = rowMajor, force: bool = false): Tensor[T]=
  ## Transform a tensor with general striding to a Tensor with contiguous layout.
  ##
  ## By default tensor will be rowMajor.
  ##
  ## By default nothing is done if the tensor is already contiguous (C Major or F major)
  ## The "force" parameter can force re-ordering to a specific layout

  if t.isContiguous and not force:
    return t
  elif t.is_C_contiguous and layout == rowMajor:
    return t
  elif t.is_F_contiguous and layout == colMajor:
    return t
  contiguousT(result, t, layout)

proc unsafeContiguous*[T](t: Tensor[T], layout: OrderType = rowMajor, force: bool = false): Tensor[T] =
  ## Transform a tensor with general striding to a Tensor with contiguous layout.
  ##
  ## If the tensor is already contiguous it is returned without copy, underlying data is shared between the input and the output.
  ##
  ## Warning ⚠:
  ##   This may be a no-copy operation with result data shared with the input.
  ##   This proc does not guarantee that a ``let`` value is immutable.
  ##
  ## By default tensor will be rowMajor.
  ##
  ## By default nothing is done if the tensor is already contiguous (C Major or F major)
  ## The "force" parameter can force re-ordering to a specific layout

  if t.isContiguous and not force:
    return t.unsafeView
  elif t.is_C_contiguous and layout == rowMajor:
    return t.unsafeView
  elif t.is_F_contiguous and layout == colMajor:
    return t.unsafeView
  contiguousT(result, t, layout)

proc reshape_with_copy[T](t: Tensor[T], new_shape: MetadataArray): Tensor[T] =
  # Can't call "tensorCpu" template here for some reason
  result = newTensorUninit[T](new_shape)
  result.apply2T(t,y)

proc reshape*(t: Tensor, new_shape: varargs[int]): Tensor =
  ## Reshape a tensor
  ##
  ## Input:
  ##   - a tensor
  ##   - a new shape. Number of elements must be the same
  ## Returns:
  ##   - a tensor with the same data but reshaped.

  let ns = new_shape.toMetadataArray
  when compileOption("boundChecks"):
    check_reshape(t, ns)

  return t.reshape_with_copy(ns)

template reshape_no_copy(t: AnyTensor, new_shape: varargs[int]): untyped =
  let ns = new_shape.toMetadataArray
  when compileOption("boundChecks"):
    check_nocopyReshape t
    check_reshape(t, ns)

  var matched_dims = 0
  for shapes in zip(t.shape, ns): # This relies on zip stopping early
    if shapes[0] != shapes[1]:
      break
    inc matched_dims

  result.shape = ns

  # Strides extended for unmatched dimension
  let ext_strides = result.shape[matched_dims..result.shape.high].shape_to_strides
  result.strides = t.strides[0..<matched_dims] & ext_strides
  result.offset = t.offset

proc unsafeReshape*(t: Tensor, new_shape: varargs[int]): Tensor =
  ## Reshape a tensor without copy.
  ##
  ## ⚠ Reshaping without copy is only possible on contiguous Tensors
  ##
  ## Warning ⚠:
  ##   This is a no-copy operation, data is shared with the input.
  ##   This proc does not guarantee that a ``let`` value is immutable.

  t.reshape_no_copy(new_shape)
  shallowCopy(result.data, t.data)

template broadcastT(t: var AnyTensor, shape: varargs[int]|MetadataArray) =
  when compileOption("boundChecks"):
    assert t.rank == shape.len

  for i in 0..<t.rank:
    if t.shape[i] == 1:
      if shape[i] != 1:
        t.shape[i] = shape[i]
        t.strides[i] = 0
    elif t.shape[i] != shape[i]:
      raise newException(ValueError, "The broadcasted size of the tensor must match existing size for non-singleton dimension")

proc broadcast*[T](t: Tensor[T], shape: varargs[int]): Tensor[T] {.noSideEffect.}=
  ## Explicitly broadcast a tensor to the specified shape.
  ##
  ## Dimension(s) of size 1 can be expanded to arbitrary size by replicating
  ## values along that dimension.
  ##
  ## Warning ⚠:
  ##   A broadcasted tensor should not be modified and only used for computation.

  result = t
  result.broadcastT(shape)

proc unsafeBroadcast*[T](t: Tensor[T], shape: varargs[int]): Tensor[T] {.noSideEffect.}=
  ## Explicitly broadcast a Tensor to the specified shape.
  ## The returned broadcasted Tensor share the underlying data with the input.
  ##
  ## Dimension(s) of size 1 can be expanded to arbitrary size by replicating
  ## values along that dimension.
  ##
  ## Warning ⚠:
  ##   This is a no-copy operation, data is shared with the input.
  ##   This proc does not guarantee that a ``let`` value is immutable.
  ##   A broadcasted tensor should not be modified and only used for computation.
  result = t.unsafeView
  result.broadcastT(shape)

proc unsafeBroadcast*[T](t: Tensor[T], shape: MetadataArray): Tensor[T] {.noSideEffect.}=
  ## Explicitly broadcast a Tensor to the specified shape.
  ## The returned broadcasted Tensor share the underlying data with the input.
  ##
  ## Dimension(s) of size 1 can be expanded to arbitrary size by replicating
  ## values along that dimension.
  ##
  ## Warning ⚠:
  ##   This is a no-copy operation, data is shared with the input.
  ##   This proc does not guarantee that a ``let`` value is immutable.
  ##   A broadcasted tensor should not be modified and only used for computation.
  result = t.unsafeView
  result.broadcastT(shape)

proc broadcast*[T: SomeNumber](val: T, shape: varargs[int]): Tensor[T] {.noSideEffect.} =
  ## Broadcast a number
  ##
  ## Input:
  ##   - a number to be broadcasted
  ##   - a tensor shape that will be broadcasted to
  ## Returns:
  ##   - a tensor with the broadcasted shape where all elements has the broadcasted value
  ##
  ## The broadcasting is made using tensor data of size 1 and 0 strides, i.e.
  ## the operation is memory efficient.
  ##
  ## Warning ⚠:
  ##   A broadcasted tensor should not be modified and only used for computation.
  ##   Modifying any value from this broadcasted tensor will change all its values.
  result.shape = shape.toMetadataArray
  # result.strides # Unneeded, autoinitialized with 0
  result.offset = 0
  result.data = newSeqWith(1, val)

template bc*(t: (Tensor|SomeNumber), shape: varargs[int]): untyped =
  ## Alias for ``broadcast``
  t.broadcast(shape)

proc unsafeBroadcast2[T](a, b: Tensor[T]): tuple[a, b: Tensor[T]] {.noSideEffect.}=
  ## Broadcast 2 tensors so they have compatible shapes for element-wise computations.
  ##
  ## Tensors in the tuple can be accessed with output.a and output.b
  ##
  ## The returned broadcasted Tensors share the underlying data with the input.
  ##
  ## Dimension(s) of size 1 can be expanded to arbitrary size by replicating
  ## values along that dimension.
  ##
  ## Warning ⚠:
  ##   This is a no-copy operation, data is shared with the input.
  ##   This proc does not guarantee that a ``let`` value is immutable.
  ##   A broadcasted tensor should not be modified and only used for computation.
  let rank = max(a.rank, b.rank)

  var shapeA, stridesA, shapeB, stridesB = newMetadataArray(rank) # initialized with 0

  for i in 0..<rank:
    let shape_A_iter = if i < rank: a.shape[i] else: 1
    let shape_B_iter = if i < rank: b.shape[i] else: 1

    if shape_A_iter == shape_B_iter:
      shapeA[i] = shape_A_iter
      shapeB[i] = shape_A_iter

      stridesA[i] = a.strides[i]
      stridesB[i] = b.strides[i]

    elif shape_A_iter == 1:
      shapeA[i] = shape_B_iter
      shapeB[i] = shape_B_iter

      # stridesA[i] is already 0
      stridesB[i] = b.strides[i]
    elif shape_B_iter == 1:
      shapeA[i] = shape_A_iter
      shapeB[i] = shape_A_iter

      stridesA[i] = a.strides[i]
      # stridesB[i] is already 0
    else:
      raise newException(ValueError, "Broadcasting error: non-singleton dimensions must be the same in both tensors")

  result.a.shape = shapeA
  result.a.strides = stridesA
  result.a.offset = a.offset
  shallowCopy(result.a.data, a.data)

  result.b.shape = shapeB
  result.b.strides = stridesB
  result.b.offset = b.offset
  shallowCopy(result.b.data, b.data)

proc exch_dim(t: Tensor, dim1, dim2: int): Tensor {.noSideEffect.}=
  if dim1 == dim2:
    return

  result = t
  swap(result.strides[dim1], result.strides[dim2])
  swap(result.shape[dim1], result.shape[dim2])

proc permute*(t: Tensor, dims: varargs[int]): Tensor {.noSideEffect.}=
  ## Permute dimensions of a tensors
  ## Input:
  ##   - a tensor
  ##   - the new dimension order
  ## Returns:
  ##   - a tensor with re-order dimension
  ## Usage:
  ##  .. code:: nim
  ##     a.permute(0,2,1) # dim 0 stays at 0, dim 1 becomes dim 2 and dim 2 becomes dim 1

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
  result = newTensorUninit[T](concat_shape)

  # Fill in the copy with the matching values
  var slices = concat_shape.mapIt((0..<it)|1)
  var iaxis = 0

  for t in t_list:
    slices[axis].a = iaxis
    slices[axis].b = iaxis + t.shape[axis] - 1
    result.slicerMut(slices, t)
    iaxis += t.shape[axis]

template squeezeT(t: var AnyTensor): untyped =
  var idx_real_dim = 0

  for i in 0..<t.rank:
    if t.shape[i] != 1:
      if i != idx_real_dim:
        t.shape[idx_real_dim] = t.shape[i]
        t.strides[idx_real_dim] = t.strides[i]
      inc idx_real_dim
  
  t.shape = t.shape[0..<idx_real_dim]
  t.strides = t.strides[0..<idx_real_dim]

  if t.rank == 0:
    t.shape.add 1
    t.strides.add 1

proc squeeze*(t: AnyTensor): AnyTensor {.noSideEffect.}=
  ## Squeeze tensors. For example a Tensor of shape @[4,1,3] will become @[4,3]
  ## Input:
  ##   - a tensor
  ## Returns:
  ##   - a tensor with singleton dimensions collapsed
  result = t
  result.squeezeT

proc unsafeSqueeze*(t: Tensor): Tensor {.noSideEffect.}=
  ## Squeeze tensors. For example a Tensor of shape @[4,1,3] will become @[4,3]
  ## Input:
  ##   - a tensor
  ## Returns:
  ##   - a tensor with singleton dimensions collapsed that share the same underlying storage
  ## Warning ⚠:
  ##   This is a no-copy operation, data is shared with the input.
  ##   This proc does not guarantee that a ``let`` value is immutable.
  result = t.unsafeView
  result.squeezeT

template squeezeT(t: var AnyTensor, axis: int): untyped =
  when compileOption("boundChecks"):
    check_squeezeAxis(t, axis)

  if t.rank > 1 and t.shape[axis] == 1: # We don't support rank 0 Tensor
    t.shape.delete(axis)
    t.strides.delete(axis)

proc squeeze*(t: Tensor, axis: int): Tensor {.noSideEffect.}=
  ## Collapse the given axis, if the dimension is not 1, it does nothing.
  ## Input:
  ##   - a tensor
  ##   - an axis (dimension)
  ## Returns:
  ##   - a tensor with that axis collapsed, if it was a singleton dimension
  result = t
  result.squeezeT(axis)

proc unsafeSqueeze*(t: Tensor, axis: int): Tensor {.noSideEffect.}=
  ## Collapse the given axis, if the dimension is not 1; it does nothing
  ## Input:
  ##   - a tensor
  ##   - an axis (dimension)
  ## Returns:
  ##   - a tensor with singleton dimensions collapsed
  ## Warning ⚠:
  ##   This is a no-copy operation, data is shared with the input.
  ##   This proc does not guarantee that a ``let`` value is immutable.
  result = t.unsafeView
  result.squeezeT(axis)

template unsqueezeT(t: var AnyTensor, axis: int): untyped =
  when compileOption("boundChecks"):
    check_unsqueezeAxis(t, axis)

  # set the stride to be consistent with the rest of the lib
  var stride: int
  if axis >= t.rank:
    stride = 1
  else:
    stride = t.shape[axis]*t.strides[axis]

  t.shape.insert(1, axis)
  t.strides.insert(stride, axis)

proc unsqueeze*(t: Tensor, axis: int): Tensor {.noSideEffect.}=
  ## Insert a new axis just before the given axis, increasing the tensor
  ## dimension (rank) by 1
  ## Input:
  ##   - a tensor
  ##   - an axis (dimension)
  ## Returns:
  ##   - a tensor with that new axis
  result = t
  result.unsqueezeT(axis)

proc unsafeUnsqueeze*(t: Tensor, axis: int): Tensor {.noSideEffect.}=
  ## Insert a new axis just before the given axis, increasing the tensor
  ## dimension (rank) by 1
  ##   - a tensor with that new axis
  ## WARNING: result share storage with input
  ## This does not guarantee `let` variable immutability
  result = t.unsafeView
  result.unsqueezeT(axis)

proc stack*[T](tensors: varargs[Tensor[T]], axis: int = 0): Tensor[T] =
  ## Join a sequence of tensors along a new axis into a new tensor.
  ## Input:
  ##   - a tensor
  ##   - an axis (dimension)
  ## Returns:
  ##   - a new stacked tensor along the new axis
  proc stack_unsqueeze(t: Tensor[T]): Tensor[T] = t.unsafeUnsqueeze(axis)
  tensors.map(stack_unsqueeze).concat(axis)
