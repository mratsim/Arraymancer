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

import  ../laser/dynamic_stack_arrays,
        ../laser/tensor/datatypes,
        nimblas, complex

export nimblas.OrderType, complex

type
  # On CPU, the tensor datastructures and basic accessors
  # are defined in laser/tensor/datatypes

  CudaStorage*[T: SomeFloat] = object
    ## Opaque seq-like structure for storage on the Cuda backend.
    ##
    ## Nim garbage collector will automatically ask cuda to clear GPU memory if data becomes unused.
    ##
    # TODO: Forward declaring this and making this completely private prevent assignment in newCudaStorage from working
    Flen*: int
    Fdata*: ptr UncheckedArray[T]
    Fref_tracking*: ref[ptr UncheckedArray[T]] # We keep ref tracking for the GC in a separate field to avoid double indirection.

  CudaTensor*[T: SomeFloat] = object
    ## Tensor data structure stored on Nvidia GPU (Cuda)
    ##   - ``shape``: Dimensions of the CudaTensor
    ##   - ``strides``: Numbers of items to skip to get the next item along a dimension.
    ##   - ``offset``: Offset to get the first item of the CudaTensor. Note: offset can be negative, in particular for slices.
    ##   - ``storage``: An opaque data storage for the CudaTensor
    ##
    ## Warning ⚠:
    ##   Assignment ```var a = b``` does not copy the data. Data modification on one CudaTensor will be reflected on the other.
    ##   However modification on metadata (shape, strides or offset) will not affect the other tensor.
    ##   Explicit copies can be made with ``clone``: ```var a = b.clone```
    shape*: MetadataArray
    strides*: MetadataArray
    offset*: int
    storage*: CudaStorage[T]

  ClStorage*[T: SomeFloat] = object
    ## Opaque seq-like structure for storage on the OpenCL backend.
    Flen*: int
    Fdata*: ptr UncheckedArray[T]
    Fref_tracking*: ref[ptr UncheckedArray[T]] # We keep ref tracking for the GC in a separate field to avoid double indirection.

  ClTensor*[T: SomeFloat] = object
    ## Tensor data structure stored on OpenCL (CPU, GPU, FPGAs or other accelerators)
    ##   - ``shape``: Dimensions of the CudaTensor
    ##   - ``strides``: Numbers of items to skip to get the next item along a dimension.
    ##   - ``offset``: Offset to get the first item of the CudaTensor. Note: offset can be negative, in particular for slices.
    ##   - ``storage``: An opaque data storage for the CudaTensor
    ##
    ## Warning ⚠:
    ##   Assignment ```var a = b``` does not copy the data. Data modification on one CudaTensor will be reflected on the other.
    ##   However modification on metadata (shape, strides or offset) will not affect the other tensor.
    ##   Explicit copies can be made with ``clone``: ```var a = b.clone```
    shape*: MetadataArray
    strides*: MetadataArray
    offset*: int
    storage*: ClStorage[T]

  AnyTensor*[T] = Tensor[T] or CudaTensor[T] or ClTensor[T]

# ###############
# Field accessors
# ###############

proc data*[T](t: Tensor[T]): seq[T] {.inline, noSideEffect, noInit.} =
  # Get tensor raw data
  # This is intended for library writer
  shallowCopy(result, t.storage.Fdata)

proc data*[T](t: var Tensor[T]): var seq[T] {.inline, noSideEffect, noInit.} =
  # Get mutable tensor raw data
  # This is intended for library writer
  shallowCopy(result, t.storage.Fdata)

proc `data=`*[T](t: var Tensor[T], s: seq[T]) {.inline, noSideEffect.}=
  # Set tensor raw data
  # This is intended for library writer
  t.storage.Fdata = s

# ################
# Tensor Metadata
# ################

proc rank*(t: AnyTensor): int {.noSideEffect, inline.}=
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

proc is_C_contiguous*(t: AnyTensor): bool {.noSideEffect, inline.}=
  ## Check if the tensor follows C convention / is row major
  var z = 1
  for i in countdown(t.shape.high,0):
    # 1. We should ignore strides on dimensions of size 1
    # 2. Strides always must have the size equal to the product of the next dimensons
    if t.shape[i] != 1 and t.strides[i] != z:
        return false
    z *= t.shape[i]
  return true

proc is_F_contiguous*(t: AnyTensor): bool {.noSideEffect, inline.}=
  ## Check if the tensor follows Fortran convention / is column major
  var z = 1
  for i in 0..<t.shape.len:
    # 1. We should ignore strides on dimensions of size 1
    # 2. Strides always must have the size equal to the product of the next dimensons
    if t.shape[i] != 1 and t.strides[i] != z:
        return false
    z *= t.shape[i]
  return true

proc isContiguous*(t: AnyTensor): bool {.noSideEffect, inline.}=
  ## Check if the tensor is contiguous
  return t.is_C_contiguous or t.is_F_contiguous

# ##################
# Raw pointer access
# ##################


proc get_data_ptr*[T](t: AnyTensor[T]): ptr T {.noSideEffect, inline.}=
  ## Input:
  ##     - A tensor
  ## Returns:
  ##     - A pointer to the real start of its data (no offset)
  unsafeAddr(t.storage.Fdata[0])

proc get_offset_ptr*[T](t: AnyTensor[T]): ptr T {.noSideEffect, inline.}=
  ## Input:
  ##     - A tensor
  ## Returns:
  ##     - A pointer to the offset start of its data
  unsafeAddr(t.storage.Fdata[t.offset])

proc dataArray*[T](t: Tensor[T]): ptr UncheckedArray[T] {.noSideEffect, inline.}=
  ## Input:
  ##     - A tensor
  ## Returns:
  ##     - A pointer to the offset start of the data.
  ##       Return value supports array indexing.
  cast[ptr UncheckedArray[T]](t.storage.Fdata[t.offset].unsafeAddr)
