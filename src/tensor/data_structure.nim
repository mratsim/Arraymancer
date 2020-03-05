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

import
  # Internal
  ../laser/dynamic_stack_arrays,
  ../laser/tensor/datatypes,
  ../private/sequninit,
  # Third-party
  nimblas,
  # Standard library
  std/[complex, typetraits]

export nimblas.OrderType, complex
export datatypes, dynamic_stack_arrays

type
  # On CPU, the tensor datastructures and basic accessors
  # are defined in laser/tensor/datatypes
  MetadataArray*{.deprecated: "Use Metadata instead".} = Metadata

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

proc data*[T](t: Tensor[T]): seq[T] {.inline, noInit, deprecated: "This used to be a way to extract raw data without copy. Use the raw pointer instead.".} =
  # Get tensor raw data
  # This is intended for library writer
  when supportsCopyMem(T):
    result = newSeqUninit[T](t.size)
    for i in 0 ..< t.size:
      result[i] = t.storage.raw_buffer[i]
  else:
    shallowCopy(result, t.storage.raw_buffer)

# TODO: pretty sure this is completely broken
proc data*[T](t: var Tensor[T]): var seq[T] {.deprecated: "This used to be a way to extract raw data without copy. Use the raw pointer instead.".} =
  # Get mutable tensor raw data
  # This is intended for library writer
  when supportsCopyMem(T):
    result = newSeqUninit(t.size)
    for i in 0 ..< t.size:
      result[i] = t.storage.raw_buffer[i]
  else:
    shallowCopy(result, t.storage.raw_buffer)

proc `data=`*[T](t: var Tensor[T], s: seq[T]) {.deprecated: "Use copyFromRaw instead".} =
  # Set tensor raw data
  # This is intended for library writer
  assert s.len > 0
  when supportsCopyMem(T):
    t.copyFromRaw(s[0].addr, s.len)
  else:
    t.storage.raw_buffer = s

# ################
# Tensor Metadata
# ################

# rank, size, is_C_contiguous defined in laser

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

proc shape_to_strides*(shape: MetadataArray, layout: OrderType = rowMajor, result: var MetadataArray) {.noSideEffect.} =
  ## Input:
  ##     - A shape (MetadataArray), for example [3,5] for a 3x5 matrix
  ##     - Optionally rowMajor (C layout - default) or colMajor (Fortran)
  ## Returns:
  ##     - The strides in C or Fortran order corresponding to this shape and layout
  ##
  ## Arraymancer defaults to rowMajor. Temporarily, CudaTensors are colMajor by default.
  # See Design document for further considerations.
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

# ##################
# Raw pointer access
# ##################

# TODO: proper getters and setters, that also update Nim refcount
#       for interoperability of Arraymancer buffers with other framework

proc get_data_ptr*[T](t: AnyTensor[T]): ptr T {.noSideEffect, inline.}=
  ## Input:
  ##     - A tensor
  ## Returns:
  ##     - A pointer to the real start of its data (no offset)
  cast[ptr T](t.storage.raw_buffer)

proc get_offset_ptr*[T](t: AnyTensor[T]): ptr T {.noSideEffect, inline.}=
  ## Input:
  ##     - A tensor
  ## Returns:
  ##     - A pointer to the offset start of its data
  t.storage.raw_buffer[t.offset].unsafeAddr

proc dataArray*[T](t: Tensor[T]): ptr UncheckedArray[T] {.noSideEffect, inline, deprecated: "Use unsafe_raw_data instead".}=
  ## Input:
  ##     - A tensor
  ## Returns:
  ##     - A pointer to the offset start of the data.
  ##       Return value supports array indexing.
  (ptr UncheckedArray[T])(t.unsafe_raw_data)
