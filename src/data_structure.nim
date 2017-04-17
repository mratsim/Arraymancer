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

    # DataKind = enum
    #    Dense
    #    Sparse

    Tensor*[B: static[Backend]; T] = object
        # Size of the datastructure is 32 bytes - perfect !
        dimensions: seq[int]
        strides: seq[int]
        offset: ptr T
        data: seq[T] # Perf note: seq are always deep copied on assignement.

template len*(t: Tensor): int = t.data.len
template shape*(t: Tensor): seq[int] = t.dimensions.reversed
template strides*(t: Tensor): seq[int] = t.strides
template rank*(t: Tensor): int = t.dimensions.len
    # 0 for scalar (unfortunately cannot be stored)
    # 1 for vector
    # 2 for matrices
    # N for N-dimension array

proc is_C_contiguous(t: Tensor): bool {.noSideEffect,inline.}=
    result = t.strides.isSorted(system.cmp[int], SortOrder.Descending)
    result = result and t.strides[t.strides.high] == 1

proc is_F_contiguous(t: Tensor): bool {.noSideEffect,inline.}=
    result = t.strides.isSorted(system.cmp[int], SortOrder.Ascending)
    result = result and t.strides[0] == 1

template offset_to_index[B,T](t: Tensor[B,T]): int =
    ## Convert the pointer offset to the corresponding integer index
    ptrMath:
        # TODO: Thoroughly test this, especially with negative offsets
        let d0: ptr T = unsafeAddr(t.data[0])
        let offset_idx: int = t.offset - d0
    offset_idx

proc `==`*[B,T](a,b: Tensor[B,T]): bool {.noSideEffect.}=
    ## Tensor comparison
    if a.dimensions != b.dimensions: return false
    elif a.strides != b.strides: return false
    elif offset_to_index(a) != offset_to_index(b): return false
    elif a.data != b.data: return false
    else: return true