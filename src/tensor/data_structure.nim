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

import  ./backend/metadataArray,
        ./backend/storage,
        nimblas

export nimblas.OrderType

type
  Backend*{.deprecated.}= enum
    ## ``Backend`` for tensor computation and memory allocation.
    ##
    ##
    ## Only deprecated procs from v0.1.3 uses this for the moment.
    Cpu,
    Cuda

  Tensor*[T] = object
    ## Tensor data structure stored on Cpu
    ##   - ``shape``: Dimensions of the tensor
    ##   - ``strides``: Numbers of items to skip to get the next item along a dimension.
    ##   - ``offset``: Offset to get the first item of the Tensor. Note: offset can be negative, in particular for slices.
    ##   - ``storage``: An "opaque" CpuStorage datatype that holds the actual data + a reference counter.
    ##                  Data is accessible via the ".data" accessor.
    ## Fields are public so that external libraries can easily construct a Tensor.
    shape*: MetadataArray
    strides*: MetadataArray
    offset*: int
    storage*: CpuStorage[T]

  CudaTensor*[T: SomeReal] = object
    ## Tensor data structure stored on Nvidia GPU (Cuda)
    ##   - ``shape``: Dimensions of the tensor
    ##   - ``strides``: Numbers of items to skip to get the next item along a dimension.
    ##   - ``offset``: Offset to get the first item of the Tensor. Note: offset can be negative, in particular for slices.
    ##   - ``data``: A cuda seq-like object that points to the data location
    ## Note: currently ``=`` assignement for CudaTensor does not copy. Both CudaTensors will share a view of the same data location.
    ## Modifying the data in one will modify the data in the other.
    ##
    ## In the future CudaTensor will leverage Nim compiler to automatically
    ## copy if a memory location would be used more than once in a mutable manner.
    shape*: MetadataArray
    strides*: MetadataArray
    offset*: int
    data*: CudaSeq[T] # Memory on Cuda device will be automatically garbage-collected

  AnyTensor*[T] = Tensor[T] or CudaTensor[T]

# #############
# Copy-on-write
# #############

proc dataFrom*[T](t: var Tensor[T], s: seq[T]) {.inline, noSideEffect.}=
  # Safely change the old storage to a new reference.
  # It relies on Nim garbage collector for cleanup when needup.
  #
  # Note: this only works without races if only the main thread can access this.
  # Also increment is only done on assignment, slices do not increment.

  var tmp_store: CpuStorage[T]
  new tmp_store

  initRef tmp_store
  tmp_store.Fdata = s

  swap(t.storage, tmp_store)

  decRef tmp_store

proc detach*[T](t: var Tensor[T]) {.inline, noSideEffect.}=
  # Create a new storage copy if more than
  # one tensor alread refer to the storage.
  if t.storage.isUniqueRef:
    return

  dataFrom(t, t.storage.Fdata)

proc `=`*[T](dst: var Tensor[T]; src: Tensor[T]) {.inline, noSideEffect.}=
  # Assignment overload to track reference count.
  # Note: only `let`, `var` and assignment to a var triggers refcounting
  # result = expression or function parameter passing will not.
  incRef src.storage
  system.`=`(dst, src)

## Use --newruntime with Arraymancer
# {.experimental.}
# proc `=destroy`*[T](c: Tensor[T]) {.inline, noSideEffect.}=
#   # Automatically called on tensor destruction. It will decrease
#   # the reference count on the shared storage
#   decRef c.storage

# ###############
# Field accessors
# ###############

proc data*[T](t: Tensor[T]): seq[T] {.inline,noInit.} =
  # Get tensor raw data
  # This is intended for library writer
  shallowCopy(result, t.storage.Fdata)

proc data*[T](t: var Tensor[T]): var seq[T] {.inline,noInit.} =
  # Get mutable tensor raw data
  # This is intended for library writer
  shallowCopy(result, t.storage.Fdata)

proc `data=`*[T](t: var Tensor[T], s: seq[T]) {.inline, noSideEffect.}=
  # Set tensor raw data
  # This is intended for library writer
  dataFrom[T](t, s)

template rank*(t: AnyTensor): int =
  ## Input:
  ##     - A tensor
  ## Returns:
  ##     - Its rank
  ##
  ##   - 0 for scalar (unfortunately cannot be stored)
  ##   - 1 for vector
  ##   - 2 for matrices
  ##   - N for N-dimension array
  t.shape.len

proc size*(t: AnyTensor): int {.noSideEffect, inline.}=
  ## Input:
  ##     - A tensor
  ## Returns:
  ##     - The total number of elements it contains
  t.shape.product

proc shape_to_strides*(shape: MetadataArray, layout: OrderType = rowMajor, result: var MetadataArray) {.noSideEffect.} =
  ## Input:
  ##     - A shape (MetadataArray), for example [3,5] for a 3x5 matrix
  ##     - Optionally rowMajor (C layout - default) or colMajor (Fortran)
  ## Returns:
  ##     - The strides in C or Fortran order corresponding to this shape and layout
  ##
  ## Arraymancer defaults to rowMajor. Temporarily, CudaTensors are colMajor by default.
  # See Design document for further considerations.
  var accum = 1
  result.len = shape.len

  if layout == rowMajor:
    for i in countdown(shape.len-1,0):
      result[i] = accum
      accum *= shape[i]
    return

  for i in 0 ..< shape.len:
    result[i] = accum
    accum *= shape[i]
  return

proc is_C_contiguous*(t: AnyTensor): bool {.noSideEffect,inline.}=
  ## Check if the tensor follows C convention / is row major
  var z = 1
  for i in countdown(t.shape.high,0):
    # 1. We should ignore strides on dimensions of size 1
    # 2. Strides always must have the size equal to the product of the next dimensons
    if t.shape[i] != 1 and t.strides[i] != z:
        return false
    z *= t.shape[i]
  return true

proc is_F_contiguous*(t: AnyTensor): bool {.noSideEffect,inline.}=
  ## Check if the tensor follows Fortran convention / is column major
  var z = 1
  for i in 0..<t.shape.len:
    # 1. We should ignore strides on dimensions of size 1
    # 2. Strides always must have the size equal to the product of the next dimensons
    if t.shape[i] != 1 and t.strides[i] != z:
        return false
    z *= t.shape[i]
  return true

proc isContiguous*(t: AnyTensor): bool {.noSideEffect,inline.}=
  ## Check if the tensor is contiguous
  t.is_C_contiguous or t.is_F_contiguous

proc get_data_ptr*[T](t: AnyTensor[T]): ptr T {.inline.}=
  ## Input:
  ##     - A tensor
  ## Returns:
  ##     - A pointer to the real start of its data (no offset)
  when t is Tensor:
    unsafeAddr(t.storage.Fdata[0])
  elif t is CudaTensor:
    unsafeAddr(t.data.data[0])

proc get_offset_ptr*[T](t: Tensor[T]): ptr T {.inline.}=
  ## Input:
  ##     - A tensor
  ## Returns:
  ##     - A pointer to the offset start of its data
  unsafeAddr(t.storage.Fdata[t.offset])

proc dataArray*[T](t: Tensor[T]): ptr UncheckedArray[T] {.inline.}=
  ## Input:
  ##     - A tensor
  ## Returns:
  ##     - A pointer to the offset start of the data.
  ##       Return value supports array indexing.
  cast[ptr UncheckedArray[T]](t.storage.Fdata[t.offset].unsafeAddr)
