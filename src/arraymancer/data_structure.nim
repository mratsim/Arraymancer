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

type
  Backend* = enum
    ## ``Backend`` for tensor computation and memory allocation.
    ##
    ## Only Cpu is supported for now.
    Cpu,
    Cuda
    # OpenCL
    # Magma

  Tensor*[T] = object
    # Size of the datastructure is 32 bytes - perfect !
    ## Tensor data structure, stored on Cpu
    ##   - ``shape``: Dimensions of the tensor
    ##   - ``strides``: Numbers of items to skip to get the next item along a dimension.
    ##   - ``offset``: Offset to get the first item of the Tensor. Note: offset can be negative, in particular for slices.
    ##   - ``data``: A sequence that holds the actual data
    shape: seq[int]
    strides: seq[int]
    offset: int
    data: seq[T] # Perf note: seq are always deep copied on "var" assignement.

  CudaTensor*[T: SomeReal] = object
    ## Tensor data structure, stored on Nvidia GPU (Cuda)
    ##   - ``shape``: Dimensions of the tensor
    ##   - ``strides``: Numbers of items to skip to get the next item along a dimension.
    ##   - ``offset``: Offset to get the first item of the Tensor. Note: offset can be negative, in particular for slices.
    ##   - ``data_ptr``: A pointer to the data location
    ##   - ``data_ref``: A reference-counted pointer to the data location
    shape: seq[int]
    strides: seq[int]
    offset: int
    data_ref: ref[ptr T] # Memory on Cuda device will be automatically garbage-collected
    len: int

  AnyTensor[T] = Tensor[T] or CudaTensor[T]

template shape*(t: AnyTensor): seq[int] =
  ## Input:
  ##     - A tensor
  ## Returns:
  ##     - Its shape
  t.shape

template strides*(t: AnyTensor): seq[int] =
  ## Input:
  ##     - A tensor
  ## Returns:
  ##     - Its strides
  t.strides

template offset*(t: AnyTensor): int =
  ## Input:
  ##     - A tensor
  ## Returns:
  ##     - Its offset
  t.offset

template rank*(t: AnyTensor): int =
  ## Input:
  ##     - A tensor
  ## Returns:
  ##     - Its shape
  ##
  ##   - 0 for scalar (unfortunately cannot be stored)
  ##   - 1 for vector
  ##   - 2 for matrices
  ##   - N for N-dimension array
  ##
  t.shape.len

proc shape_to_strides(shape: seq[int]): seq[int] {.noSideEffect.} =
  ## Compute strides matching with dimensions.
  return (shape & 1)[1..shape.len].scanr(a * b)

proc is_C_contiguous(t: AnyTensor): bool {.noSideEffect.}=
  ## Check if C convention / Row Major
  result = t.strides == t.shape.shape_to_strides
  result = result and t.strides[t.strides.high] == 1

proc is_F_contiguous(t: AnyTensor): bool {.noSideEffect.}=
  ## Check if Fortran convention / Column Major
  result = t.strides.reversed == t.shape.reversed.shape_to_strides
  result = result and t.strides[0] == 1

proc isContiguous(t: AnyTensor): bool {.noSideEffect.}=
  return t.is_C_contiguous or t.is_F_contiguous

proc getTransposeTarget(t: AnyTensor): TransposeType {.noSideEffect.}=
  ## TransposeType is introduced by ``nimblas``
  ## Default layout is Row major.
  ## Everytime it is worth it or fused with a BLAS operation we change the strides to Row-Major
  if is_C_contiguous(t): return TransposeType.noTranspose
  elif is_F_contiguous(t): return TransposeType.transpose
  else: raise newException(ValueError,"Operation not supported for this matrix. It has a non-contiguous layout")

template get_data_ptr*[T](t: AnyTensor[T]): ptr T =
  ## Input:
  ##     - A tensor
  ## Returns:
  ##     - A pointer to the start of its data
  when t is Tensor:
    unsafeAddr(t.data[0])
  elif t is CudaTensor:
    t.data_ref[]