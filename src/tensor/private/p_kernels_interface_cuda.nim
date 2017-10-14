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

import  ../backend/cuda,
        ../backend/global_config,
        ../data_structure

# Design: due to limitations and difficulties in C++
# (no mutable iterators, stack hidden from GC, exception issue, casting workaround for builtin_assume_aligned)
# I choose to sacrifice readability here to avoid a plethora of "when not defined(cpp) in the codebase.
# This file will serve as a bridge between Cuda C++ and Nim C.

# Explanation:
#
# - Cuda C++ code is handled by p_kernels_cuda
# - p_kernels_cuda is compiled separately as a C++ file to avoid polluting the whole project with C++ semantics
# - The compiled C++ is renamed to .cu to make NVCC happy
# - We tell Nim not to forget to compile/link to this .cu

# #########################################################
# Compiling cuda kernels separately

const nimCliArgs = "nim e script_kernels_interface_cuda.nims"
const execCmd {.used.}= staticExec(nimCliArgs,"","")

static:
  echo  " ### Begin: cuda kernels compilation ###\n" &
        execCmd &
        "\n ### End: cuda kernels compilation ###"


# Tell Nim/NVCC to compile and link against the kernels file
# We go from "./src/tensor/private" to "./nimcache"
{.compile: "../../../nimcache/arraymancer_p_kernels_cuda.cu".}


# #########################################################
# # Call assignements, copy and in-place operations kernels

template cuda_assign_call*[T: SomeReal](
  kernel_name: untyped, destination: var CudaTensor[T], source: CudaTensor[T]): untyped =
  ## Does the heavy-lifting to format the tensors for the cuda call
  # TODO: why doesn't this template works with the global

  let dst = layoutOnDevice destination
  let src = layoutOnDevice source

  kernel_name(
    CUDA_HOF_TPB, CUDA_HOF_BPG,
    src.rank, dst.len, # Note: small shortcut, in this case len and size are the same
    dst.shape[], dst.strides[],
    dst.offset, dst.data,
    src.shape[], src.strides[],
    src.offset, src.data
  )

# ##################################################
# # Call binary operations kernels

template cuda_binary_call*[T: SomeReal](
  kernel_name: untyped, destination: var CudaTensor[T], a, b: CudaTensor[T]): untyped =
  ## Does the heavy-lifting to format the tensors for the cuda call
  # TODO: why doesn't this template works with the global

  let dst = layoutOnDevice destination
  let src_a = layoutOnDevice a
  let src_b = layoutOnDevice b

  kernel_name(
    CUDA_HOF_TPB, CUDA_HOF_BPG,
    src_a.rank, dst.len, # Note: small shortcut, in this case len and size are the same
    dst.shape[], dst.strides[],
    dst.offset, dst.data,
    src_a.shape[], src_a.strides[],
    src_a.offset, src_a.data,
    src_b.shape[], src_b.strides[],
    src_b.offset, src_b.data
  )

###########################

proc cuda_mAdd_f32*(
    blocksPerGrid, threadsPerBlock: cint,
    rank, len: cint,
    dst_shape, dst_strides: ptr cint, dst_offset: cint, dst_data: ptr float32,
    src_shape, src_strides: ptr cint, src_offset: cint, src_data: ptr float32
  ) {.importc.}

proc cuda_mAdd*(
    blocksPerGrid, threadsPerBlock: cint,
    rank, len: cint,
    dst_shape, dst_strides: ptr cint, dst_offset: cint, dst_data: ptr float32,
    src_shape, src_strides: ptr cint, src_offset: cint, src_data: ptr float32
  ) =
  cuda_mAdd_f32(blocksPerGrid, threadsPerBlock, rank, len,
  dst_shape, dst_strides, dst_offset, dst_data,
  src_shape, src_strides, src_offset, src_data)

proc cuda_mAdd_f64*(
    blocksPerGrid, threadsPerBlock: cint,
    rank, len: cint,
    dst_shape, dst_strides: ptr cint, dst_offset: cint, dst_data: ptr float64,
    src_shape, src_strides: ptr cint, src_offset: cint, src_data: ptr float64
  ) {.importc.}

proc cuda_mAdd*(
    blocksPerGrid, threadsPerBlock: cint,
    rank, len: cint,
    dst_shape, dst_strides: ptr cint, dst_offset: cint, dst_data: ptr float64,
    src_shape, src_strides: ptr cint, src_offset: cint, src_data: ptr float64
  ) =
  cuda_mAdd_f64(blocksPerGrid, threadsPerBlock, rank, len,
  dst_shape, dst_strides, dst_offset, dst_data,
  src_shape, src_strides, src_offset, src_data)


############################################
proc cuda_mSub_f32*(
    blocksPerGrid, threadsPerBlock: cint,
    rank, len: cint,
    dst_shape, dst_strides: ptr cint, dst_offset: cint, dst_data: ptr float32,
    src_shape, src_strides: ptr cint, src_offset: cint, src_data: ptr float32
  ) {.importc.}

proc cuda_mSub*(
    blocksPerGrid, threadsPerBlock: cint,
    rank, len: cint,
    dst_shape, dst_strides: ptr cint, dst_offset: cint, dst_data: ptr float32,
    src_shape, src_strides: ptr cint, src_offset: cint, src_data: ptr float32
  ) =
  cuda_mSub_f32(blocksPerGrid, threadsPerBlock, rank, len,
  dst_shape, dst_strides, dst_offset, dst_data,
  src_shape, src_strides, src_offset, src_data)

proc cuda_mSub_f64*(
    blocksPerGrid, threadsPerBlock: cint,
    rank, len: cint,
    dst_shape, dst_strides: ptr cint, dst_offset: cint, dst_data: ptr float64,
    src_shape, src_strides: ptr cint, src_offset: cint, src_data: ptr float64
  ) {.importc.}

proc cuda_mSub*(
    blocksPerGrid, threadsPerBlock: cint,
    rank, len: cint,
    dst_shape, dst_strides: ptr cint, dst_offset: cint, dst_data: ptr float64,
    src_shape, src_strides: ptr cint, src_offset: cint, src_data: ptr float64
  ) =
  cuda_mSub_f64(blocksPerGrid, threadsPerBlock, rank, len,
  dst_shape, dst_strides, dst_offset, dst_data,
  src_shape, src_strides, src_offset, src_data)

##############################################
proc cuda_unsafeContiguous_f32*(
    blocksPerGrid, threadsPerBlock: cint,
    rank, len: cint,
    dst_shape, dst_strides: ptr cint, dst_offset: cint, dst_data: ptr float32,
    src_shape, src_strides: ptr cint, src_offset: cint, src_data: ptr float32
  ) {.importc.}

proc cuda_unsafeContiguous*(
    blocksPerGrid, threadsPerBlock: cint,
    rank, len: cint,
    dst_shape, dst_strides: ptr cint, dst_offset: cint, dst_data: ptr float32,
    src_shape, src_strides: ptr cint, src_offset: cint, src_data: ptr float32
  ) =
  cuda_unsafeContiguous_f32(blocksPerGrid, threadsPerBlock, rank, len,
  dst_shape, dst_strides, dst_offset, dst_data,
  src_shape, src_strides, src_offset, src_data)

proc cuda_unsafeContiguous_f64*(
    blocksPerGrid, threadsPerBlock: cint,
    rank, len: cint,
    dst_shape, dst_strides: ptr cint, dst_offset: cint, dst_data: ptr float64,
    src_shape, src_strides: ptr cint, src_offset: cint, src_data: ptr float64
  ) {.importc.}

proc cuda_unsafeContiguous*(
    blocksPerGrid, threadsPerBlock: cint,
    rank, len: cint,
    dst_shape, dst_strides: ptr cint, dst_offset: cint, dst_data: ptr float64,
    src_shape, src_strides: ptr cint, src_offset: cint, src_data: ptr float64
  ) =
  cuda_unsafeContiguous_f64(blocksPerGrid, threadsPerBlock, rank, len,
  dst_shape, dst_strides, dst_offset, dst_data,
  src_shape, src_strides, src_offset, src_data)


################################################

proc cuda_Add_f32*(
    blocksPerGrid, threadsPerBlock: cint,
    rank, len: cint,
    dst_shape, dst_strides: ptr cint, dst_offset: cint, dst_data: ptr float32,
    a_shape, a_strides: ptr cint, a_offset: cint, a_data: ptr float32,
    b_shape, b_strides: ptr cint, b_offset: cint, b_data: ptr float32
  ) {.importc.}

proc cuda_Add*(
    blocksPerGrid, threadsPerBlock: cint,
    rank, len: cint,
    dst_shape, dst_strides: ptr cint, dst_offset: cint, dst_data: ptr float32,
    a_shape, a_strides: ptr cint, a_offset: cint, a_data: ptr float32,
    b_shape, b_strides: ptr cint, b_offset: cint, b_data: ptr float32
  ) =
  cuda_Add_f32(
    blocksPerGrid, threadsPerBlock,
    rank, len,
    dst_shape, dst_strides, dst_offset, dst_data,
    a_shape, a_strides, a_offset, a_data,
    b_shape, b_strides, b_offset, b_data
  )


proc cuda_Add_f64*(
    blocksPerGrid, threadsPerBlock: cint,
    rank, len: cint,
    dst_shape, dst_strides: ptr cint, dst_offset: cint, dst_data: ptr float64,
    a_shape, a_strides: ptr cint, a_offset: cint, a_data: ptr float64,
    b_shape, b_strides: ptr cint, b_offset: cint, b_data: ptr float64
  ) {.importc.}

proc cuda_Add*(
    blocksPerGrid, threadsPerBlock: cint,
    rank, len: cint,
    dst_shape, dst_strides: ptr cint, dst_offset: cint, dst_data: ptr float64,
    a_shape, a_strides: ptr cint, a_offset: cint, a_data: ptr float64,
    b_shape, b_strides: ptr cint, b_offset: cint, b_data: ptr float64
  ) =
  cuda_Add_f64(
    blocksPerGrid, threadsPerBlock,
    rank, len,
    dst_shape, dst_strides, dst_offset, dst_data,
    a_shape, a_strides, a_offset, a_data,
    b_shape, b_strides, b_offset, b_data
  )

############################################
proc cuda_Sub_f32*(
    blocksPerGrid, threadsPerBlock: cint,
    rank, len: cint,
    dst_shape, dst_strides: ptr cint, dst_offset: cint, dst_data: ptr float32,
    a_shape, a_strides: ptr cint, a_offset: cint, a_data: ptr float32,
    b_shape, b_strides: ptr cint, b_offset: cint, b_data: ptr float32
  ) {.importc.}

proc cuda_Sub*(
    blocksPerGrid, threadsPerBlock: cint,
    rank, len: cint,
    dst_shape, dst_strides: ptr cint, dst_offset: cint, dst_data: ptr float32,
    a_shape, a_strides: ptr cint, a_offset: cint, a_data: ptr float32,
    b_shape, b_strides: ptr cint, b_offset: cint, b_data: ptr float32
  ) =
  cuda_Sub_f32(
    blocksPerGrid, threadsPerBlock,
    rank, len,
    dst_shape, dst_strides, dst_offset, dst_data,
    a_shape, a_strides, a_offset, a_data,
    b_shape, b_strides, b_offset, b_data
  )

proc cuda_Sub_f64*(
    blocksPerGrid, threadsPerBlock: cint,
    rank, len: cint,
    dst_shape, dst_strides: ptr cint, dst_offset: cint, dst_data: ptr float64,
    a_shape, a_strides: ptr cint, a_offset: cint, a_data: ptr float64,
    b_shape, b_strides: ptr cint, b_offset: cint, b_data: ptr float64
  ) {.importc.}

proc cuda_Sub*(
    blocksPerGrid, threadsPerBlock: cint,
    rank, len: cint,
    dst_shape, dst_strides: ptr cint, dst_offset: cint, dst_data: ptr float64,
    a_shape, a_strides: ptr cint, a_offset: cint, a_data: ptr float64,
    b_shape, b_strides: ptr cint, b_offset: cint, b_data: ptr float64
  ) =
  cuda_Sub_f64(
    blocksPerGrid, threadsPerBlock,
    rank, len,
    dst_shape, dst_strides, dst_offset, dst_data,
    a_shape, a_strides, a_offset, a_data,
    b_shape, b_strides, b_offset, b_data
  )