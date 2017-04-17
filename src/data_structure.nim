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

    # StrideKind = enum
    # Either C convention or Fortran convention are needed for BLAS optimization for Tensor of Rank 1 or 2
    #    C_contiguous # Row Major - Default. Last index is the fastest changing (columns in 2D, depth in 3D) - Rows (slowest), Columns, Depth (fastest)
    #    F_contiguous # Col Major. First index is the fastest changing (rows in 2D, depth in 3D) - Rows (fastest), Columns, Depth (slowest)
    #    Universal # Any stride
    # For deep learning on images, depth represents colors channels and change the fastest, rows represent another image in a batch and change the slowest.
    # Hence C convention is the best.

    # DataKind = enum
    #    Dense
    #    Sparse

    Tensor*[B: static[Backend]; T] = object
        # N is the rank of the Tensor.
        # 0 for scalar (unfortunately cannot be stored)
        # 1 for vector
        # 2 for matrices
        # N for N-dimension array
        # Size of the datastructure is 32 bytes - perfect !
        #
        dimensions: seq[int]
        strides: seq[int]
        offset: ptr T # Should annote `not nil` but due to pointer arithmetic that cannot be proven
        data: seq[T] # Perf note: seq are always deep copied on assignement

template len*(t: Tensor): int = t.data.len
template shape*(t: Tensor): seq[int] = t.dimensions.reversed
template strides*(t: Tensor): seq[int] = t.strides
template rank*(t: Tensor): int = t.dimensions.len

proc is_C_contiguous(t: Tensor): bool {.noSideEffect,inline.}=
    result = t.strides.isSorted(system.cmp[int], SortOrder.Descending)
    result = result and t.strides[t.strides.high] == 1

proc is_F_contiguous(t: Tensor): bool {.noSideEffect,inline.}=
    result = t.strides.isSorted(system.cmp[int], SortOrder.Ascending)
    result = result and t.strides[0] == 1

template offset_to_index[B,T](t: Tensor[B,T]): int =
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