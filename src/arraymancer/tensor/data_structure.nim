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
  ../laser/dynamic_stack_arrays,
  ../laser/tensor/datatypes,
  nimblas,
  # Standard library
  std/[complex]

export nimblas.OrderType, complex
export datatypes, dynamic_stack_arrays

type
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
    shape*: Metadata
    strides*: Metadata
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
    shape*: Metadata
    strides*: Metadata
    offset*: int
    storage*: ClStorage[T]

  AnyTensor*[T] = Tensor[T] or CudaTensor[T] or ClTensor[T]

# ###############
# Field accessors
# ###############

proc `data=`*[T](t: var Tensor[T], s: seq[T]) {.deprecated: "Use copyFromRaw instead".} =
  # Set tensor raw data
  # This is intended for library writer
  assert s.len > 0
  when T is KnownSupportsCopyMem:
    t.copyFromRaw(s[0].unsafeAddr, s.len)
  else:
    t.storage.raw_buffer = s

# ################
# Tensor Metadata
# ################

func rank*[T](t: CudaTensor[T] or ClTensor[T]): range[0 .. LASER_MAXRANK] {.inline.} =
  t.shape.len

func size*[T](t: CudaTensor[T] or ClTensor[T]): Natural {.inline.} =
  t.shape.product

proc shape_to_strides*(shape: Metadata, layout: OrderType = rowMajor, result: var Metadata) {.noSideEffect.} =
  ## Input:
  ##     - A shape (Metadata), for example [3,5] for a 3x5 matrix
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

func is_C_contiguous*(t: CudaTensor or ClTensor): bool =
  ## Check if the tensor follows C convention / is row major
  var cur_size = 1
  for i in countdown(t.rank - 1,0):
    # 1. We should ignore strides on dimensions of size 1
    # 2. Strides always must have the size equal to the product of the next dimensions
    if t.shape[i] != 1 and t.strides[i] != cur_size:
        return false
    cur_size *= t.shape[i]
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


proc get_data_ptr*[T: KnownSupportsCopyMem](t: Tensor[T]): ptr T {.noSideEffect, inline.}=
  ## Input:
  ##     - A tensor
  ## Returns:
  ##     - A pointer to the real start of its data (no offset)
  cast[ptr T](t.storage.raw_buffer)

proc get_data_ptr*[T: not KnownSupportsCopyMem](t: AnyTensor[T]): ptr T {.error: "`get_data_ptr`" &
  " cannot be safely used for GC'ed types!".}

proc get_offset_ptr*[T: KnownSupportsCopyMem](t: Tensor[T]): ptr T {.noSideEffect, inline.}=
  ## Input:
  ##     - A tensor
  ## Returns:
  ##     - A pointer to the offset start of its data
  t.storage.raw_buffer[t.offset].unsafeAddr

proc get_offset_ptr*[T: not KnownSupportsCopyMem](t: AnyTensor[T]): ptr T {.error: "`get_offset_ptr`" &
  " cannot be safely used for GC'ed types!".}

proc get_data_ptr*[T](t: CudaTensor[T] or ClTensor[T]): ptr T {.noSideEffect, inline.}=
  ## Input:
  ##     - A tensor
  ## Returns:
  ##     - A pointer to the real start of its data (no offset)
  cast[ptr T](t.storage.Fdata)

proc get_offset_ptr*[T](t: CudaTensor[T] or ClTensor[T]): ptr T {.noSideEffect, inline.}=
  ## Input:
  ##     - A tensor
  ## Returns:
  ##     - A pointer to the offset start of its data
  t.storage.Fdata[t.offset].unsafeAddr

proc dataArray*[T: KnownSupportsCopyMem](t: Tensor[T]): ptr UncheckedArray[T] {.noSideEffect, inline, deprecated: "Use toUnsafeView instead".}=
  ## Input:
  ##     - A tensor
  ## Returns:
  ##     - A pointer to the offset start of the data.
  ##       Return value supports array indexing.
  cast[ptr UncheckedArray[T]](t.unsafe_raw_offset[t.offset].unsafeAddr)

proc dataArray*[T: not KnownSupportsCopyMem](t: Tensor[T]): ptr UncheckedArray[T] {.error: "`dataArray` " &
  " is deprecated for mem copyable types and not supported for GC'ed types!".}
