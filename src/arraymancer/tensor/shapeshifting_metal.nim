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

import  ./data_structure,
        ./private/p_init_metal,
        ./private/p_shapeshifting,
        ./init_metal,
        ./backend/metal/metal_backend

proc transpose*(t: MetalTensor): MetalTensor {.noSideEffect.}=
  ## Transpose a MetalTensor.
  ##
  ## For N-d Tensor with shape (0, 1, 2 ... n-1) the resulting tensor will have shape (n-1, ... 2, 1, 0)
  t.shape.reversed(result.shape)
  t.strides.reversed(result.strides)
  result.offset = t.offset
  result.storage = t.storage
  result.cpuData = t.cpuData

proc isContiguous*(t: MetalTensor): bool {.inline.} =
  ## Check if the MetalTensor is contiguous in memory
  ## A tensor is contiguous if its strides match either row-major or col-major layout
  if t.rank == 0:
    return true
  
  # Check for row-major (C-style) contiguous
  var expectedStride = 1
  var isRowMajor = true
  for i in countdown(t.rank - 1, 0):
    if t.strides[i] != expectedStride:
      isRowMajor = false
      break
    expectedStride *= t.shape[i]
  
  if isRowMajor:
    return true
  
  # Check for col-major (Fortran-style) contiguous
  expectedStride = 1
  var isColMajor = true
  for i in 0 ..< t.rank:
    if t.strides[i] != expectedStride:
      isColMajor = false
      break
    expectedStride *= t.shape[i]
  
  return isColMajor

proc asContiguous*[T: SomeFloat](t: MetalTensor[T], layout: OrderType = rowMajor, force: bool = false): MetalTensor[T] {.noinit.} =
  ## Return a contiguous copy of the MetalTensor using GPU kernel
  if not force and t.isContiguous:
    return t
  
  # Create new contiguous tensor
  result = newMetalTensor[T](t.shape)
  
  # Convert shape and strides to seq for GPU kernel
  var shapeSeq = newSeq[int](t.rank)
  var stridesSeq = newSeq[int](t.rank)
  for i in 0 ..< t.rank:
    shapeSeq[i] = t.shape[i]
    stridesSeq[i] = t.strides[i]
  
  # Use GPU kernel to copy data with arbitrary strides to contiguous buffer
  metalContiguousCopy[T](
    t.storage.Fbuffer,
    result.storage.Fbuffer,
    shapeSeq,
    stridesSeq,
    t.storage.Flen
  )

proc reshape*(t: MetalTensor, new_shape: varargs[int]): MetalTensor =
  ## Reshape a MetalTensor without copy.
  ##
  ## ⚠ Reshaping without copy is only possible on contiguous rowMajor Tensors
  
  reshape_no_copy(t, new_shape, result, rowMajor)
  result.storage = t.storage
  result.cpuData = t.cpuData

proc broadcast*(t: MetalTensor, shape: varargs[int]): MetalTensor {.noSideEffect.}=
  ## Broadcast a MetalTensor to a new shape.
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

proc broadcast*(t: MetalTensor, shape: Metadata): MetalTensor {.noSideEffect.}=
  ## Broadcast a MetalTensor to a new shape.
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

proc broadcast2*[T](a, b: MetalTensor[T]): tuple[a, b: MetalTensor[T]] {.noSideEffect.}=
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
