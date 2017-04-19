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
        Cpu
        # Cuda
        # OpenCL
        # Magma

    Tensor*[B: static[Backend]; T] = object
        # Size of the datastructure is 32 bytes - perfect !
        dimensions: seq[int]
        strides: seq[int]
        offset: int
        data: seq[T] # Perf note: seq are always deep copied on "var" assignement.

template len*(t: Tensor): int = t.data.len
template shape*(t: Tensor): seq[int] = t.dimensions.reversed
template strides*(t: Tensor): seq[int] = t.strides
template rank*(t: Tensor): int = t.dimensions.len
    # 0 for scalar (unfortunately cannot be stored)
    # 1 for vector
    # 2 for matrices
    # N for N-dimension array

proc shape_to_strides(shape: seq[int]): seq[int] {.noSideEffect,inline.} =
    return (shape & 1)[1..shape.len].scanr(a * b)

proc is_C_contiguous(t: Tensor): bool {.noSideEffect,inline.}=
    result = t.strides == t.shape.shape_to_strides
    result = result and t.strides[t.strides.high] == 1

proc is_F_contiguous(t: Tensor): bool {.noSideEffect,inline.}=
    result = t.strides.reversed == t.dimensions.shape_to_strides
    result = result and t.strides[0] == 1

proc `==`*[B,T](a,b: Tensor[B,T]): bool {.noSideEffect.}=
    ## Tensor comparison
    if a.dimensions != b.dimensions: return false
    elif a.strides != b.strides: return false
    elif a.offset != b.offset: return false
    elif a.data != b.data: return false
    else: return true

## Get a pointer to the start of the data. Needed for BLAS.
template get_data_ptr[B,T](t: Tensor[B,T]): ptr T = unsafeAddr(t.data[0])