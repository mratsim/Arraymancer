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

import ./private/p_kernels_interface_cuda,
       ./private/p_init_cuda,
       ./private/p_shapeshifting,
       ./data_structure

include ./private/incl_accessors_cuda,
        ./private/incl_higher_order_cuda,
        ./private/incl_kernels_cuda

proc transpose*(t: CudaTensor): CudaTensor {.noSideEffect.}=
  ## Transpose a Tensor.
  ##
  ## For N-d Tensor with shape (0, 1, 2 ... n-1) the resulting tensor will have shape (n-1, ... 2, 1, 0)

  t.shape.reversed(result.shape)
  t.strides.reversed(result.strides)
  result.offset = t.offset
  result.storage = t.storage

cuda_assign_glue("cuda_asContiguous", "CopyOp", cuda_asContiguous)

proc asContiguous*[T: SomeFloat](t: CudaTensor[T], layout: OrderType = rowMajor, force: bool = false):
  CudaTensor[T] {.noSideEffect, error: "NOT WORKING RIGHT NOW TODO: FIX".}=
  ## Transform a tensor with general striding to a Tensor with contiguous layout.
  ##
  ## By default CudaTensor will be colMajor (contrary to a cpu tensor).
  ##
  ## By default nothing is done if the tensor is already contiguous (C Major or F major)
  ## The "force" parameter can force re-ordering to a specific layout
  # TODO: fix. this proc always outputs rowmajor, no matter the input.
  # probably has to do with all the cuda tensors being colmajor by default,
  # plus probably some double-negative of two bugs making the other procs work.

  if t.isContiguous and not force:
    return t
  elif t.is_F_contiguous and layout == colMajor:
    return t
  elif t.is_C_contiguous and layout == rowMajor:
    return t

  result = newCudaTensor[T](t.shape, layout)

  cuda_assign_call(cuda_asContiguous, result, t)


proc reshape*(t: CudaTensor, new_shape: varargs[int]): CudaTensor =
  ## Reshape a CudaTensor without copy.
  ##
  ## ⚠ Reshaping without copy is only possible on contiguous rowMajor Tensors

  t.reshape_no_copy(new_shape, result, rowMajor)
  result.storage = t.storage

proc broadcast*(t: CudaTensor, shape: varargs[int]): CudaTensor {.noSideEffect.}=
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
  result.broadcastImpl(shape)

proc broadcast*(t: CudaTensor, shape: Metadata): CudaTensor {.noSideEffect.}=
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
  result.broadcastImpl(shape)

proc broadcast2*[T](a, b: CudaTensor[T]): tuple[a, b: CudaTensor[T]] {.noSideEffect.}=
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

proc squeeze*(t: CudaTensor, axis: int): CudaTensor {.noSideEffect.}=
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
  result.squeezeImpl(axis)

proc unsqueeze*(t: CudaTensor, axis: int): CudaTensor {.noSideEffect.}=
  ## Insert a new axis just before the given axis, increasing the CudaTensor
  ## dimension (rank) by 1
  ##   - a tensor with that new axis
  ## Warning ⚠:
  ##   This is a no-copy operation, data is shared with the input.
  ##   This proc does not guarantee that a ``let`` value is immutable.
  result = t
  result.unsqueezeImpl(axis)
