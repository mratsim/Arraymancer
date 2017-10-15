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

import ./backend/metadataArray,
       ./private/p_kernels_interface_cuda,
       ./private/p_init_cuda,
       ./private/p_shapeshifting,
       ./data_structure

include ./private/incl_accessors_cuda,
        ./private/incl_higher_order_cuda,
        ./private/incl_kernels_cuda

proc unsafeTranspose*(t: CudaTensor): CudaTensor {.noSideEffect.}=
  ## Transpose a Tensor.
  ##
  ## For N-d Tensor with shape (0, 1, 2 ... n-1) the resulting tensor will have shape (n-1, ... 2, 1, 0)
  ##
  ## Warning ⚠ CudaTensor temporary default:
  ##   This is a no-copy operation, data is shared with the input.
  ##   This proc does not guarantee that a ``let`` value is immutable.

  t.shape.reversed(result.shape)
  t.strides.reversed(result.strides)
  result.offset = t.offset
  result.data = t.data

cuda_assign_glue("cuda_unsafeContiguous", "CopyOp", cuda_unsafeContiguous)

proc unsafeContiguous*[T: SomeReal](t: CudaTensor[T], layout: OrderType = colMajor, force: bool = false):
  CudaTensor[T] {.noSideEffect.}=
  ## Transform a tensor with general striding to a Tensor with contiguous layout.
  ##
  ## By default CudaTensor will be colMajor (contrary to a cpu tensor).
  ##
  ## By default nothing is done if the tensor is already contiguous (C Major or F major)
  ## The "force" parameter can force re-ordering to a specific layout
  ##
  ## Warning ⚠ CudaTensor temporary default:
  ##   If the CudaTensor is contiguous, this is a no-copy operation, data is shared with the input.
  ##   This proc does not guarantee that a ``let`` value is immutable.

  if t.isContiguous and not force:
    return t
  elif t.is_F_contiguous and layout == colMajor:
    return t
  elif t.is_C_contiguous and layout == rowMajor:
    return t

  result = newCudaTensor[T](t.shape, layout)

  cuda_assign_call(cuda_unsafeContiguous, result, t)


proc unsafeReshape*(t: CudaTensor, new_shape: varargs[int]): CudaTensor =
  ## Reshape a CudaTensor without copy.
  ##
  ## ⚠ Reshaping without copy is only possible on contiguous Tensors
  ##
  ## Warning ⚠:
  ##   This is a no-copy operation, data is shared with the input.
  ##   This proc does not guarantee that a ``let`` value is immutable.

  t.reshape_no_copy(new_shape)
  result.data = t.data

proc unsafeBroadcast*(t: CudaTensor, shape: varargs[int]): CudaTensor {.noSideEffect.}=
  ## Explicitly broadcast a CudaTensor to the specified shape.
  ## The returned broadcasted CudaTensor share the underlying data with the input.
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

proc unsafeBroadcast*(t: CudaTensor, shape: MetadataArray): CudaTensor {.noSideEffect.}=
  ## Explicitly broadcast a CudaTensor to the specified shape.
  ## The returned broadcasted CudaTensor share the underlying data with the input.
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

proc unsafeBroadcast2*[T](a, b: CudaTensor[T]): tuple[a, b: CudaTensor[T]] {.noSideEffect.}=
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

  result.a.data = a.data
  result.b.data = b.data

proc unsafeSqueeze*(t: CudaTensor, axis: int): CudaTensor {.noSideEffect.}=
  ## Collapse the given axis, if the dimension is not 1; it does nothing
  ## Input:
  ##   - a CudaTensor
  ##   - an axis (dimension)
  ## Returns:
  ##   - a CudaTensor with singleton dimensions collapsed
  ## Warning ⚠:
  ##   This is a no-copy operation, data is shared with the input.
  ##   This proc does not guarantee that a ``let`` value is immutable.
  result = t
  result.squeezeT(axis)

proc unsafeUnsqueeze*(t: CudaTensor, axis: int): CudaTensor {.noSideEffect.}=
  ## Insert a new axis just before the given axis, increasing the CudaTensor
  ## dimension (rank) by 1
  ##   - a tensor with that new axis
  ## Warning ⚠:
  ##   This is a no-copy operation, data is shared with the input.
  ##   This proc does not guarantee that a ``let`` value is immutable.
  result = t
  result.unsqueezeT(axis)