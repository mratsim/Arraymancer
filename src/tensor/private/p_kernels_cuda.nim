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

import ./p_kernels_interface_cuda

include incl_accessors_cuda,
        incl_kernels_cuda,
        incl_higher_order_cuda

# Design: due to limitations and difficulties in C++
# (no mutable iterators, stack hidden from GC, exception issue, casting workaround for builtin_assume_aligned)
# I choose to sacrifice readability here to avoid a plethora of "when not defined(cpp) in the codebase.
# This file will serve as a bridge between Cuda C++ and Nim C.

# Explanation:
#
# - To avoid creating a proc AST from scratch: we create a dummy proc.
# - pass it to an overloading template (cuda_assign_glue for in-place/assignments)
# - overloading template creates the cuda code, the import and a Nim wrapper
# - Unfortunately the Nim wrapper cannot be called as-is in other file as it uses C++ semantics
# - So we add an indirection that will be exported.

proc cuda_inPlaceAdd_cpp = discard # This is a hack so that the symbol is open
cuda_assign_glue(cuda_inPlaceAdd_cpp, "InPlaceAddOp")

proc cuda_inPlaceAdd*[T: SomeReal](
    blocksPerGrid, threadsPerBlock: cint,
    rank, len: cint,
    dst_shape, dst_strides: ptr cint, dst_offset: cint, dst_data: ptr T,
    src_shape, src_strides: ptr cint, src_offset: cint, src_data: ptr T
  ) =
  cuda_inPlaceAdd_cpp[T](blocksPerGrid, threadsPerBlock, rank, len,
  dst_shape, dst_strides, dst_offset, dst_data,
  src_shape, src_strides, src_offset, src_data)

proc cuda_inPlaceSub_cpp = discard # This is a hack so that the symbol is open
cuda_assign_glue(cuda_inPlaceSub_cpp, "InPlaceSubOp")

proc cuda_inPlaceSub*[T: SomeReal](
    blocksPerGrid, threadsPerBlock: cint,
    rank, len: cint,
    dst_shape, dst_strides: ptr cint, dst_offset: cint, dst_data: ptr T,
    src_shape, src_strides: ptr cint, src_offset: cint, src_data: ptr T
  ) =
  cuda_inPlaceSub_cpp[T](blocksPerGrid, threadsPerBlock, rank, len,
  dst_shape, dst_strides, dst_offset, dst_data,
  src_shape, src_strides, src_offset, src_data)

proc cuda_unsafeContiguous_cpp = discard # This is a hack so that the symbol is open
cuda_assign_glue(cuda_unsafeContiguous_cpp, "CopyOp")

proc cuda_unsafeContiguous*[T: SomeReal](
    blocksPerGrid, threadsPerBlock: cint,
    rank, len: cint,
    dst_shape, dst_strides: ptr cint, dst_offset: cint, dst_data: ptr T,
    src_shape, src_strides: ptr cint, src_offset: cint, src_data: ptr T
  ) =
  cuda_unsafeContiguous_cpp[T](blocksPerGrid, threadsPerBlock, rank, len,
  dst_shape, dst_strides, dst_offset, dst_data,
  src_shape, src_strides, src_offset, src_data)

proc cuda_Add_cpp = discard # This is a hack so that the symbol is open
cuda_binary_glue(cuda_Add_cpp, "AddOp")

proc cuda_Add*[T: SomeReal](
    blocksPerGrid, threadsPerBlock: cint,
    rank, len: cint,
    dst_shape, dst_strides: ptr cint, dst_offset: cint, dst_data: ptr T,
    a_shape, a_strides: ptr cint, a_offset: cint, a_data: ptr T,
    b_shape, b_strides: ptr cint, b_offset: cint, b_data: ptr T
  ) =
  cuda_Add_cpp[T](
    blocksPerGrid, threadsPerBlock,
    rank, len,
    dst_shape, dst_strides, dst_offset, dst_data,
    a_shape, a_strides, a_offset, a_data,
    b_shape, b_strides, b_offset, b_data
  )

proc cuda_Sub_cpp = discard # This is a hack so that the symbol is open
cuda_binary_glue(cuda_Sub_cpp, "SubOp")

proc cuda_Sub*[T: SomeReal](
    blocksPerGrid, threadsPerBlock: cint,
    rank, len: cint,
    dst_shape, dst_strides: ptr cint, dst_offset: cint, dst_data: ptr T,
    a_shape, a_strides: ptr cint, a_offset: cint, a_data: ptr T,
    b_shape, b_strides: ptr cint, b_offset: cint, b_data: ptr T
  ) =
  cuda_Sub_cpp[T](
    blocksPerGrid, threadsPerBlock,
    rank, len,
    dst_shape, dst_strides, dst_offset, dst_data,
    a_shape, a_strides, a_offset, a_data,
    b_shape, b_strides, b_offset, b_data
  )