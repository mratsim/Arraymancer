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
        ./data_structure, ./init_cpu, ./higher_order_applymap,
        sequtils

# NOTE: Procs that accepts shape are duplicated to accept both varargs and MetadataArray
# until either https://github.com/nim-lang/Nim/issues/6528 or https://github.com/nim-lang/Nim/issues/6529
# are solved/added.

proc transpose*(t: Tensor): Tensor {.noInit,noSideEffect,inline.} =
  ## Transpose a Tensor.
  ##
  ## For N-d Tensor with shape (0, 1, 2 ... n-1) the resulting tensor will have shape (n-1, ... 2, 1, 0)
  ##
  ## Data is not copied or modified, only metadata is modified.
  t.shape.reversed(result.shape)
  t.strides.reversed(result.strides)
  result.offset = t.offset
  result.storage = t.storage

proc asContiguous*[T](t: Tensor[T], layout: OrderType = rowMajor, force: bool = false): Tensor[T] {.noInit.} =
  ## Transform a tensor with general striding to a Tensor with contiguous layout.
  ##
  ## By default tensor will be rowMajor.
  ##
  ## The layout is kept if the tensor is already contiguous (C Major or F major)
  ## The "force" parameter can force re-ordering to a specific layout.
  ##
  ## Result is always a fully packed tensor even if the input is a contiguous slice.

  let cCont = t.is_C_contiguous
  let fCont = t.is_F_contiguous

  if (cCont or fCont) and not force:
    return t
  elif cCont and layout == rowMajor:
    return t
  elif fCont and layout == colMajor:
    return t
  contiguousImpl(t, layout, result)

proc reshape*(t: Tensor, new_shape: varargs[int]): Tensor {.noInit.} =
  ## Reshape a tensor
  ##
  ## Input:
  ##   - a tensor
  ##   - a new shape. Number of elements must be the same
  ## Returns:
  ##   - a tensor with the same data but reshaped.
  reshapeImpl(t, new_shape, result)

proc reshape*(t: Tensor, new_shape: MetadataArray): Tensor {.noInit.} =
  ## Reshape a tensor
  ##
  ## Input:
  ##   - a tensor
  ##   - a new shape. Number of elements must be the same
  ## Returns:
  ##   - a tensor with the same data but reshaped.
  reshapeImpl(t, new_shape, result)

proc broadcast*[T](t: Tensor[T], shape: varargs[int]): Tensor[T] {.noInit,noSideEffect.}=
  ## Explicitly broadcast a tensor to the specified shape.
  ##
  ## Dimension(s) of size 1 can be expanded to arbitrary size by replicating
  ## values along that dimension.
  ##
  ## Warning ⚠:
  ##   A broadcasted tensor should not be modified and only used for computation.

  result = t
  result.broadcastImpl(shape)

proc broadcast*[T](t: Tensor[T], shape: MetadataArray): Tensor[T] {.noInit,noSideEffect.}=
  ## Explicitly broadcast a tensor to the specified shape.
  ##
  ## Dimension(s) of size 1 can be expanded to arbitrary size by replicating
  ## values along that dimension.
  ##
  ## Warning ⚠:
  ##   A broadcasted tensor should not be modified and only used for computation.

  result = t
  result.broadcastImpl(shape)

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
  result.data = @[val]

proc broadcast*[T: SomeNumber](val: T, shape: MetadataArray): Tensor[T] {.noInit,noSideEffect.} =
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
  result.data = @[val]

template bc*(t: (Tensor|SomeNumber), shape: varargs[int]): untyped =
  ## Alias for ``broadcast``
  t.broadcast(shape)

template bc*(t: (Tensor|SomeNumber), shape: MetadataArray): untyped =
  ## Alias for ``broadcast``
  t.broadcast(shape)

proc broadcast2*[T](a, b: Tensor[T]): tuple[a, b: Tensor[T]] {.noSideEffect, noInit.}=
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

  broadcast2Impl(a,b, result)

  result.a.storage = a.storage
  result.b.storage = b.storage


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
  result = t
  permuteImpl(result, dims)

proc concat*[T](t_list: varargs[Tensor[T]], axis: int): Tensor[T]  {.noInit.}=
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

func squeeze*(t: AnyTensor): AnyTensor {.noInit.}=
  ## Squeeze tensors. For example a Tensor of shape [4,1,3] will become [4,3]
  ## Input:
  ##   - a tensor
  ## Returns:
  ##   - a tensor with singleton dimensions collapsed
  result = t
  result.squeezeImpl

func squeeze*(t: Tensor, axis: int): Tensor {.noInit.}=
  ## Collapse the given axis, if the dimension is not 1, it does nothing.
  ## Input:
  ##   - a tensor
  ##   - an axis (dimension)
  ## Returns:
  ##   - a tensor with that axis collapsed, if it was a singleton dimension
  result = t
  result.squeezeImpl(axis)

func unsqueeze*(t: Tensor, axis: int): Tensor {.noInit.}=
  ## Insert a new axis just before the given axis, increasing the tensor
  ## dimension (rank) by 1
  ## Input:
  ##   - a tensor
  ##   - an axis (dimension)
  ## Returns:
  ##   - a tensor with that new axis
  result = t
  result.unsqueezeImpl(axis)

proc stack*[T](tensors: varargs[Tensor[T]], axis = 0): Tensor[T] {.noInit.} =
  ## Join a sequence of tensors along a new axis into a new tensor.
  ## Input:
  ##   - a tensor
  ##   - an axis (dimension)
  ## Returns:
  ##   - a new stacked tensor along the new axis
  func stack_unsqueeze(t: Tensor[T]): Tensor[T] = t.unsqueeze(axis)
  tensors.map(stack_unsqueeze).concat(axis)
