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


# Auto-generate glue between Nim functions, cuda higher-order functions and basic op

# ##################################################
# # Assignements, copy and in-place operations
template cuda_assign_glue(
  kernel_name: untyped, op_name: string): untyped =
  # Kernel_name must be an open symbol
  # As a hack to avoid building an AST macro with new call
  # declare a dummy proc with `proc kernel_name = discard`

  {.emit:["""
  template<typename T>
  inline void """,kernel_name.astToStr,"""(
    const int blocksPerGrid, const int threadsPerBlock,
    const int rank,
    const int len,
    const int * __restrict__ dst_shape,
    const int * __restrict__ dst_strides,
    const int dst_offset,
    T * __restrict__ dst_data,
    const int * __restrict__ src_shape,
    const int * __restrict__ src_strides,
    const int src_offset,
    const T * __restrict__ src_data){

      cuda_apply2<<<blocksPerGrid, threadsPerBlock>>>(
        rank, len,
        dst_shape, dst_strides, dst_offset, dst_data,
        """,op_name,"""<T>(),
        src_shape, src_strides, src_offset, src_data
      );
    }
    """].}

  const import_string:string = kernel_name.astToStr & "<'*8>(@)"
  # We pass the 8th parameter type to the template.
  # The "*" in '*8 is needed to remove the pointer *

  proc kernel_name[T: SomeReal](
    blocksPerGrid, threadsPerBlock: cint,
    rank, len: cint,
    dst_shape, dst_strides: ptr cint, dst_offset: cint, dst_data: ptr T,
    src_shape, src_strides: ptr cint, src_offset: cint, src_data: ptr T
  ) {.importcpp: import_string, noSideEffect.}

template cuda_assign_call[T: SomeReal](
  kernel_name: untyped, destination: var CudaTensor[T], source: CudaTensor[T]): untyped =
  ## Does the heavy-lifting to format the tensors for the cuda call
  # TODO: why doesn't this template works with the global 

  let dst = destination.layoutOnDevice
  let src = source.layoutOnDevice

  kernel_name(
    CUDA_HOF_TPB, CUDA_HOF_BPG,
    src.rank, dst.len, # Note: small shortcut, in this case len and size are the same
    dst.shape[], dst.strides[],
    dst.offset, dst.data,
    src.shape[], src.strides[],
    src.offset, src.data
  )

# ##################################################
# # Binary operations
template cuda_binary_glue(
  kernel_name: untyped, op_name: string): untyped =
  # Kernel_name must be an open symbol
  # As a hack to avoid building an AST macro with new call
  # declare a dummy proc with `proc kernel_name = discard`

  # TODO: optimize number of args to reduce register pressure

  {.emit:["""
  template<typename T>
  inline void """,kernel_name.astToStr,"""(
    const int blocksPerGrid, const int threadsPerBlock,
    const int rank,
    const int len,
    const int * __restrict__ dst_shape,
    const int * __restrict__ dst_strides,
    const int dst_offset,
    T * __restrict__ dst_data,
    const int * __restrict__ A_shape,
    const int * __restrict__ A_strides,
    const int A_offset,
    const T * __restrict__ A_data,
    const int * __restrict__ B_shape,
    const int * __restrict__ B_strides,
    const int B_offset,
    const T * __restrict__ B_data){

      cuda_apply3<<<blocksPerGrid, threadsPerBlock>>>(
        rank, len,
        dst_shape, dst_strides, dst_offset, dst_data,
        A_shape, A_strides, A_offset, A_data,
        """,op_name,"""<T>(),
        B_shape, B_strides, B_offset, B_data
      );
    }
    """].}

  const import_string:string = kernel_name.astToStr & "<'*8>(@)"
  # We pass the 8th parameter type to the template.
  # The "*" in '*8 is needed to remove the pointer *

  proc kernel_name[T: SomeReal](
    blocksPerGrid, threadsPerBlock: cint,
    rank, len: cint,
    dst_shape, dst_strides: ptr cint, dst_offset: cint, dst_data: ptr T,
    a_shape, a_strides: ptr cint, a_offset: cint, a_data: ptr T,
    b_shape, b_strides: ptr cint, b_offset: cint, b_data: ptr T
  ) {.importcpp: import_string, noSideEffect.}

template cuda_binary_call[T: SomeReal](
  kernel_name: untyped, destination: var CudaTensor[T], a, b: CudaTensor[T]): untyped =
  ## Does the heavy-lifting to format the tensors for the cuda call
  # TODO: why doesn't this template works with the global

  let dst = destination.layoutOnDevice
  let src_a = a.layoutOnDevice
  let src_b = b.layoutOnDevice

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