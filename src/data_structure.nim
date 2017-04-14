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
    # Either RowMajor, ColMajor or Contiguous are needed for BLAS optimization for Tensor of Rank 1 or 2
    #    RowMajor # C convention. Stride for row (the last) is 1.
    #    ColMajor # Fortran convention. Stride for column (the first) is 1.
    #    Contiguous # For Vector/classic arrays
    #    Universal # Any stride

    Tensor*[B: static[Backend]; T] = object
        # N is the rank of the Tensor.
        # 0 for scalar (unfortunately cannot be stored)
        # 1 for vector
        # 2 for matrices
        # N for N-dimension array
        # Size of the datastructure is 32 bytes - perfect !
        #
        dimsizes: seq[int]
        strides: seq[int]
        offset: int # To be changed to ptr T to avoids bounds checking when iterating over the Tensor?
        data: seq[T] # Perf note: seq are always deep copied on assignement
        #
        # Open design question: should the rank of the Tensor be part of its type signature?
        # This would allow us to use array instead of seq
        # Otherwise to have dimsizes and strides on the stack and limit GC we would need VLAs
        # Another alternative are unchecked arrays (of uint8? to save on size, and optimize cache lines)

template len*(t: Tensor): int = t.data.len
template dim*(t: Tensor): seq[int] = t.dimsizes
template rank*(t: Tensor): int = t.dimsizes.high
template shape*(t: Tensor): int = t.dimsizes

proc newTensor*(dim: seq[int], T: typedesc, B: static[Backend]): Tensor[B,T] =
    # Compute strides matching with dimensions.
    # Row-Major ordering, rows have strides of 1
    let strides = (dim & 1)[1..dim.len].scanr(a * b)

    ##scanr
    result.dimsizes = dim
    result.strides = strides
    result.data = newSeq[T](dim.foldl(a * b))
    result.offset = 0 # addr tmp.data[0]
    result