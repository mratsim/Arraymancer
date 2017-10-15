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

import  ./backend/metadataArray,
        ./private/p_shapeshifting,
        ./private/p_checks,
        ./private/p_accessors_macros_write,
        ./data_structure, ./init_cpu, ./higher_order,
        nimblas, sequtils

proc transpose*(t: Tensor): Tensor {.noInit,noSideEffect,inline.} =
  ## Transpose a Tensor.
  ##
  ## For N-d Tensor with shape (0, 1, 2 ... n-1) the resulting tensor will have shape (n-1, ... 2, 1, 0)
  ##
  ## Data is copied as-is and not modified.
  t.shape.reversed(result.shape)
  t.strides.reversed(result.strides)
  result.offset = t.offset
  result.data = t.data


proc unsafeTranspose*(t: Tensor): Tensor {.noInit,noSideEffect,inline.} =
  ## Transpose a Tensor without copy.
  ##
  ## Warning ⚠:
  ##   This is a no-copy operation, data is shared with the input.
  ##   This proc does not guarantee that a ``let`` value is immutable.
  ##
  ## For N-d Tensor with shape (0, 1, 2 ... n-1) the resulting tensor will have shape (n-1, ... 2, 1, 0)
  t.shape.reversed(result.shape)
  t.strides.reversed(result.strides)
  result.offset = t.offset
  shallowCopy(result.data, t.data)

proc asContiguous*[T](t: Tensor[T], layout: OrderType = rowMajor, force: bool = false): Tensor[T] {.noInit.} =
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

proc unsafeContiguous*[T](t: Tensor[T], layout: OrderType = rowMajor, force: bool = false): Tensor[T] {.noInit.} =
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

proc reshape*(t: Tensor, new_shape: varargs[int]): Tensor {.noInit.} =
  ## Reshape a tensor
  ##
  ## Input:
  ##   - a tensor
  ##   - a new shape. Number of elements must be the same
  ## Returns:
  ##   - a tensor with the same data but reshaped.

  when compileOption("boundChecks"):
    check_reshape(t, new_shape.toMetadataArray)

  return t.reshape_with_copy(new_shape)

proc unsafeReshape*(t: Tensor, new_shape: varargs[int]): Tensor {.noInit.} =
  ## Reshape a tensor without copy.
  ##
  ## ⚠ Reshaping without copy is only possible on contiguous Tensors
  ##
  ## Warning ⚠:
  ##   This is a no-copy operation, data is shared with the input.
  ##   This proc does not guarantee that a ``let`` value is immutable.

  t.reshape_no_copy(new_shape)
  shallowCopy(result.data, t.data)


proc broadcast*[T](t: Tensor[T], shape: varargs[int]): Tensor[T] {.noInit,noSideEffect.}=
  ## Explicitly broadcast a tensor to the specified shape.
  ##
  ## Dimension(s) of size 1 can be expanded to arbitrary size by replicating
  ## values along that dimension.
  ##
  ## Warning ⚠:
  ##   A broadcasted tensor should not be modified and only used for computation.

  result = t
  result.broadcastT(shape)

proc unsafeBroadcast*[T](t: Tensor[T], shape: varargs[int]): Tensor[T] {.noInit,noSideEffect.}=
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

proc unsafeBroadcast*[T](t: Tensor[T], shape: MetadataArray): Tensor[T] {.noInit,noSideEffect.}=
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

proc broadcast*[T: SomeNumber](val: T, shape: varargs[int]): Tensor[T] {.noInit,noSideEffect.} =
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
  result.shape.copyFrom(shape)
  # result.strides # Unneeded, autoinitialized with 0
  result.offset = 0
  result.data = newSeqWith(1, val)

template bc*(t: (Tensor|SomeNumber), shape: varargs[int]): untyped =
  ## Alias for ``broadcast``
  t.broadcast(shape)

proc unsafeBroadcast2*[T](a, b: Tensor[T]): tuple[a, b: Tensor[T]] {.noSideEffect.}=
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

  broadcast2T(a,b, result)

  shallowCopy(result.a.data, a.data)
  shallowCopy(result.b.data, b.data)

proc permute*(t: Tensor, dims: varargs[int]): Tensor {.noInit,noSideEffect.}=
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


proc concat*[T](t_list: varargs[Tensor[T]], axis: int): Tensor[T]  {.noInit,noSideEffect.}=
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

  let concat_shape = t0.shape[0..<axis] & axis_dim & t0.shape[axis+1..<t0.shape.len]

  ## Setup the Tensor
  result = newTensorUninit[T](concat_shape)

  ## Fill in the copy with the matching values
  ### First a sequence of SteppedSlices corresponding to each tensors to concatenate
  var slices = concat_shape.mapIt((0..<it)|1)
  var iaxis = 0

  ### Now, create "windows" in the result tensor and assign corresponding tensors
  for t in t_list:
    slices[axis].a = iaxis
    slices[axis].b = iaxis + t.shape[axis] - 1
    result.slicerMut(slices, t)
    iaxis += t.shape[axis]

proc squeeze*(t: AnyTensor): AnyTensor {.noInit,noSideEffect.}=
  ## Squeeze tensors. For example a Tensor of shape @[4,1,3] will become @[4,3]
  ## Input:
  ##   - a tensor
  ## Returns:
  ##   - a tensor with singleton dimensions collapsed
  result = t
  result.squeezeT

proc unsafeSqueeze*(t: Tensor): Tensor {.noInit,noSideEffect.}=
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

proc squeeze*(t: Tensor, axis: int): Tensor {.noInit,noSideEffect.}=
  ## Collapse the given axis, if the dimension is not 1, it does nothing.
  ## Input:
  ##   - a tensor
  ##   - an axis (dimension)
  ## Returns:
  ##   - a tensor with that axis collapsed, if it was a singleton dimension
  result = t
  result.squeezeT(axis)

proc unsafeSqueeze*(t: Tensor, axis: int): Tensor {.noInit,noSideEffect.}=
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

proc unsqueeze*(t: Tensor, axis: int): Tensor {.noInit,noSideEffect.}=
  ## Insert a new axis just before the given axis, increasing the tensor
  ## dimension (rank) by 1
  ## Input:
  ##   - a tensor
  ##   - an axis (dimension)
  ## Returns:
  ##   - a tensor with that new axis
  result = t
  result.unsqueezeT(axis)

proc unsafeUnsqueeze*(t: Tensor, axis: int): Tensor {.noInit,noSideEffect.}=
  ## Insert a new axis just before the given axis, increasing the tensor
  ## dimension (rank) by 1
  ##   - a tensor with that new axis
  ## WARNING: result share storage with input
  ## This does not guarantee `let` variable immutability
  result = t.unsafeView
  result.unsqueezeT(axis)

proc stack*[T](tensors: varargs[Tensor[T]], axis: int = 0): Tensor[T] {.noInit.} =
  ## Join a sequence of tensors along a new axis into a new tensor.
  ## Input:
  ##   - a tensor
  ##   - an axis (dimension)
  ## Returns:
  ##   - a new stacked tensor along the new axis
  proc stack_unsqueeze(t: Tensor[T]): Tensor[T] = t.unsafeUnsqueeze(axis)
  tensors.map(stack_unsqueeze).concat(axis)
