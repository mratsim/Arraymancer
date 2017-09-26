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
    ##
    ## Only deprecated procs from v0.1.3 uses this for the moment.
    Cpu,
    Cuda

  Tensor*[T] = object
    # Size of the datastructure is 32 bytes - perfect !
    ## Tensor data structure stored on Cpu
    ##   - ``shape``: Dimensions of the tensor
    ##   - ``strides``: Numbers of items to skip to get the next item along a dimension.
    ##   - ``offset``: Offset to get the first item of the Tensor. Note: offset can be negative, in particular for slices.
    ##   - ``data``: A sequence that holds the actual data
    ## Fields are public so that external libraries can easily construct a Tensor.
    shape*: seq[int]
    strides*: seq[int]
    offset*: int
    data*: seq[T] # Perf note: seq are always deep copied on "var" assignement.

  CudaSeq* [T: SomeReal] = object
    ## Seq-like structure on the Cuda backend.
    ##
    ## Nim garbage collector will automatically ask cuda to clear GPU memory if ``data`` becomes unused.
    len: int
    data: ref[ptr UncheckedArray[T]]

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
    shape*: seq[int]
    strides*: seq[int]
    offset*: int
    data*: CudaSeq[T] # Memory on Cuda device will be automatically garbage-collected

  AnyTensor[T] = Tensor[T] or CudaTensor[T]

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

proc shape_to_strides*(shape: seq[int], layout: OrderType = rowMajor): seq[int] {.noSideEffect.} =
  ## Input:
  ##     - A shape (seq of int), for example @[3,5] for a 3x5 matrix
  ##     - Optionally rowMajor (C layout - default) or colMajor (Fortran)
  ## Returns:
  ##     - The strides in C or Fortran order corresponding to this shape and layout
  ##
  ## Arraymancer defaults to rowMajor. Temporarily, CudaTensors are colMajor by default.
  # See Design document for further considerations.
  if layout == rowMajor:
    return (shape & 1)[1..shape.len].scanr(a * b)

  return (1 & shape)[0..shape.high].scanl(a * b)

proc is_C_contiguous*(t: AnyTensor): bool {.noSideEffect,inline.}=
  ## Check if the tensor follows C convention / is row major
  if not (t.strides == t.shape.shape_to_strides()):
    return false
  return (t.strides[t.strides.high] == 1)

proc is_F_contiguous*(t: AnyTensor): bool {.noSideEffect,inline.}=
  ## Check if the tensor follows Fortran convention / is column major
  if not (t.strides == t.shape.shape_to_strides(colMajor)):
    return false
  return (t.strides[0] == 1)

proc isContiguous*(t: AnyTensor): bool {.noSideEffect,inline.}=
  ## Check if the tensor is contiguous
  return t.is_C_contiguous or t.is_F_contiguous

proc isFullyIterable(t: AnyTensor): bool {.noSideEffect,inline.}=
  return t.data.len == t.size

proc isFullyIterableAs(t1: AnyTensor, t2: AnyTensor): bool {.noSideEffect,inline.}=
  let datasize = t1.data.len
  return (t1.strides == t2.strides) and
         (t1.data.len == t2.data.len) and
         (t1.size == datasize) and
         (t2.size == datasize)

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
    unsafeAddr(t.data.data[0])