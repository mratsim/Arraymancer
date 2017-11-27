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

import  ../backend/metadataArray,
        ../private/p_shapeshifting,
        ../private/p_checks,
        ../private/p_accessors_macros_write,
        ../data_structure, ../init_cpu, ../higher_order


proc unsafeTranspose*(t: Tensor): Tensor {.noInit,noSideEffect,inline, deprecated.} =
  ## DEPRECATED
  ##
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

proc unsafeContiguous*[T](t: Tensor[T], layout: OrderType = rowMajor, force: bool = false): Tensor[T] {.noInit, deprecated.} =
  ## DEPRECATED
  ##
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

  let cCont = t.is_C_contiguous
  let fCont = t.is_F_contiguous

  if (cCont or fCont) and not force:
    return t
  elif cCont and layout == rowMajor:
    return t
  elif fCont and layout == colMajor:
    return t
  contiguousT(result, t, layout)

proc unsafeReshape*(t: Tensor, new_shape: varargs[int]): Tensor {.noInit, deprecated.} =
  ## DEPRECATED
  ##
  ## Reshape a tensor without copy.
  ##
  ## ⚠ Reshaping without copy is only possible on contiguous Tensors
  ##
  ## Warning ⚠:
  ##   This is a no-copy operation, data is shared with the input.
  ##   This proc does not guarantee that a ``let`` value is immutable.

  t.reshape_no_copy(new_shape, result)
  result.storage = t.storage

proc unsafeReshape*(t: Tensor, new_shape: MetadataArray): Tensor {.noInit, deprecated.} =
  ## DEPRECATED
  ##
  ## Reshape a tensor without copy.
  ##
  ## ⚠ Reshaping without copy is only possible on contiguous Tensors
  ##
  ## Warning ⚠:
  ##   This is a no-copy operation, data is shared with the input.
  ##   This proc does not guarantee that a ``let`` value is immutable.

  t.reshape_no_copy(new_shape, result)
  result.storage = t.storage


proc unsafeBroadcast*[T](t: Tensor[T], shape: varargs[int]): Tensor[T] {.noInit,noSideEffect, deprecated.}=
  ## DEPRECATED
  ##
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
  result = t
  result.broadcastT(shape)

proc unsafeBroadcast*[T](t: Tensor[T], shape: MetadataArray): Tensor[T] {.noInit,noSideEffect, deprecated.}=
  ## DEPRECATED
  ##
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
  result = t
  result.broadcastT(shape)

proc unsafeBroadcast2*[T](a, b: Tensor[T]): tuple[a, b: Tensor[T]] {.noSideEffect, deprecated.}=
  ## DEPRECATED
  ##
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


proc unsafePermute*(t: Tensor, dims: varargs[int]): Tensor {.noInit,noSideEffect, deprecated.}=
  ## DEPRECATED
  ##
  ## Permute dimensions of a tensors
  ## Input:
  ##   - a tensor
  ##   - the new dimension order
  ## Returns:
  ##   - a tensor with re-order dimension
  ## Usage:
  ##  .. code:: nim
  ##     a.permute(0,2,1) # dim 0 stays at 0, dim 1 becomes dim 2 and dim 2 becomes dim 1
  ##
  ## Warning ⚠:
  ##   This is a no-copy operation, data is shared with the input.
  ##   This proc does not guarantee that a ``let`` value is immutable.
  ##   A broadcasted tensor should not be modified and only used for computation.

  # TODO: bounds check
  result = t
  permuteT(result, dims)

proc unsafeSqueeze*(t: Tensor): Tensor {.noInit,noSideEffect, deprecated.}=
  ## DEPRECATED
  ##
  ## Squeeze tensors. For example a Tensor of shape [4,1,3] will become [4,3]
  ## Input:
  ##   - a tensor
  ## Returns:
  ##   - a tensor with singleton dimensions collapsed that share the same underlying storage
  ## Warning ⚠:
  ##   This is a no-copy operation, data is shared with the input.
  ##   This proc does not guarantee that a ``let`` value is immutable.
  result = t
  result.squeezeT

proc unsafeSqueeze*(t: Tensor, axis: int): Tensor {.noInit,noSideEffect, deprecated.}=
  ## DEPRECATED
  ##
  ## Collapse the given axis, if the dimension is not 1; it does nothing
  ## Input:
  ##   - a tensor
  ##   - an axis (dimension)
  ## Returns:
  ##   - a tensor with singleton dimensions collapsed
  ## Warning ⚠:
  ##   This is a no-copy operation, data is shared with the input.
  ##   This proc does not guarantee that a ``let`` value is immutable.
  result = t
  result.squeezeT(axis)

proc unsafeUnsqueeze*(t: Tensor, axis: int): Tensor {.noInit,noSideEffect, deprecated.}=
  ## DEPRECATED
  ##
  ## Insert a new axis just before the given axis, increasing the tensor
  ## dimension (rank) by 1
  ##   - a tensor with that new axis
  ## WARNING: result share storage with input
  ## This does not guarantee `let` variable immutability
  result = t
  result.unsqueezeT(axis)