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

proc transpose*(t: CudaTensor): CudaTensor {.noSideEffect.}=
  ## Transpose a Tensor.
  ##
  ## For N-d Tensor with shape (0, 1, 2 ... n-1) the resulting tensor will have shape (n-1, ... 2, 1, 0)
  ##
  ## Data is copied as-is and not modified.
  ## WARNING: The input and output tensors share the underlying data storage on GPU
  ## Modification of data will affect both

  result.shape = t.shape.reversed
  result.strides = t.strides.reversed
  result.offset = t.offset
  result.data_ref = t.data_ref
  result.len = t.len


{.emit:["""
  template<typename T>
  inline void cuda_asContiguous(
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
        CopyOp<T>(),
        src_shape, src_strides, src_offset, src_data
      );
    }
  """].}

proc cuda_asContiguous[T: SomeReal](
  blocksPerGrid, threadsPerBlock: cint,
  rank, len: cint,
  dst_shape, dst_strides: ptr cint, dst_offset: cint, dst_data: ptr T,
  src_shape, src_strides: ptr cint, src_offset: cint, src_data: ptr T
) {.importcpp: "cuda_asContiguous<'*8>(@)", noSideEffect.}
# We pass the 8th parameter type to the template.
# The "*" in '*8 is needed to remove the pointer *

proc asContiguous*[T: SomeReal](t: CudaTensor[T], layout: OrderType = colMajor, force: bool = false):
  CudaTensor[T] {.noSideEffect.}=
  ## Transform a tensor with general striding to a Tensor with contiguous layout.
  ## By default CudaTensor will be colMajor (contrary to a cpu tensor).
  ## By default nothing is done if the tensor is already contiguous (C Major or F major)
  ## The "force" parameter can force re-ordering to a specific layout
  ##
  ## WARNING, until optimized value semantics are implemented, this returns a tensor that shares
  ## the underlying data with the original IF it was contiguous.

  if t.isContiguous and not force:
    return t
  elif t.is_F_contiguous and layout == colMajor:
    return t
  elif t.is_C_contiguous and layout == rowMajor:
    return t

  result = newCudaTensor[T](t.shape, layout)

  let dst = result.layoutOnDevice
  let src = t.layoutOnDevice

  cuda_asContiguous(
    CUDA_HOF_TPB, CUDA_HOF_BPG,
    src.rank, dst.len, # Note: small shortcut, in this case len and size (shape.product) are the same
    dst.shape[], dst.strides[],
    dst.offset, dst.data,
    src.shape[], src.strides[],
    src.offset, src.data
  )