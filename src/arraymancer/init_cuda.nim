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


proc cudaMalloc[T](size: int): ptr T {.noSideEffect.}=
  ## Internal proc.
  ## Wrap CudaMAlloc(var pointer, size) -> Error_code
  let s = size * sizeof(T)
  check cudaMalloc(cast[ptr pointer](addr result), s)

proc deallocCuda[T](p: ref[ptr T]) {.noSideEffect.}=
  if not p[].isNil:
    check cudaFree(p[])

proc shallowCopy*[T](t: var CudaTensor[T]): CudaTensor[T] {.inline,noSideEffect.}=
  ## Input:
  ##     - A ``var`` tensor
  ## Returns:
  ##     - A shallow copy.
  ##
  ## WARNING !
  ##   Both tensors shares the same memory. Data modification on one will be reflected on the other.
  ##   However modifying the shape, strides or offset will not affect the other.
  
  # shape and strides fields have value semantics by default
  # data_ref has ref semantics
  system.`=`(result, t)

proc clone*[T](t: CudaTensor[T]): CudaTensor[T] =
  ## Clone (deep copy) a CudaTensor.
  ## Tensor is copied as is.
  ##
  ## For example it will not be made contiguous.
  ## Use `asContiguous` for this case

  # Note: due to modifying the defaultStream global var for async memcopy
  # proc cannot be tagged noSideEffect

  new(result.data_ref, deallocCuda)
  result.shape = t.shape
  result.strides = t.strides
  result.offset = t.offset
  result.len = t.len
  result.data_ref[] = cudaMalloc[T](result.len)
  let size = result.len * sizeof(T)

  check cudaMemCpyAsync(result.get_data_ptr,
                        t.get_data_ptr,
                        size,
                        cudaMemcpyDeviceToDevice,
                        defaultStream) # defaultStream is a cudaStream_t global var

# ###########################################################
# Implement value semantics for CudaTensor
# Pending https://github.com/nim-lang/Nim/issues/6348
# Tracked in https://github.com/mratsim/Arraymancer/issues/19
#
# proc `=`*[T](dest: var CudaTensor[T]; src: CudaTensor[T]) =
#   ## Overloading the assignment operator
#   ## It will have value semantics by default
#   new(dest.data_ref, deallocCuda)
#   dest.shape = src.shape
#   dest.strides = src.strides
#   dest.offset = src.offset
#   dest.len = src.len
#   dest.data_ref[] = cudaMalloc[T](dest.len)
#
#   let size = dest.len * sizeof(T)
#
#   check cudaMemCpy(dest.get_data_ptr,
#                    src.get_data_ptr,
#                    size,
#                    cudaMemcpyDeviceToDevice)
#   echo "Value copied"
#
# proc `=`*[T](dest: var CudaTensor[T]; src: CudaTensor[T]{call}) {.inline.}=
#   ## Overloading the assignment operator
#   ## Optimized version that knows that
#   ## the source CudaTensor is unique and thus don't need to be copied
#   system.`=`(result, t)
#   echo "Value moved"

proc newCudaTensor[T: SomeReal](shape: openarray[int], layout: OrderType = colMajor): CudaTensor[T] {.noSideEffect.}=
  ## Internal proc
  ## Allocate a CudaTensor
  ## WARNING: The Cuda memory is not initialized to 0

  # TODO: default to RowMajor. Pending https://github.com/mratsim/Arraymancer/issues/22
  # As mentionned in design doc, an element-wise kernel will avoid relying on CuBLAS
  # for inplace operation that requires column major layout.

  new(result.data_ref, deallocCuda)
  result.shape = @shape
  result.len = result.shape.product
  result.data_ref[] = cudaMalloc[T](result.len)
  result.strides = shape_to_strides(result.shape, layout)
  result.offset = 0

proc cuda*[T:SomeReal](t: Tensor[T]): CudaTensor[T] =
  ## Convert a tensor on Cpu to a tensor on a Cuda device.
  # Note: due to modifying the defaultStream global var for async copy
  # proc cannot be tagged noSideEffect

  result = newCudaTensor[T](t.shape)

  # TODO: avoid reordering rowMajor tensors. This is only needed for inplace operation in CUBLAS.
  let contig_t = t.asContiguous(colMajor, force = true)
  let size = result.len * sizeof(T)

  # For host to device we use non-blocking copy
  # Host can proceed with computation.
  # On CUDA device, next operations will be batch in the stream queue.
  check cudaMemCpyAsync(result.get_data_ptr,
                        contig_t.get_data_ptr,
                        size,
                        cudaMemcpyHostToDevice,
                        defaultStream) # defaultStream is a cudaStream_t global var

proc cpu*[T:SomeReal](t: CudaTensor[T]): Tensor[T] {.noSideEffect.}=
  ## Convert a tensor on a Cuda device to a tensor on Cpu.
  # We use blocking copy in this case to make sure
  # all data is available for future computation

  result.shape = t.shape
  result.strides = t.strides
  result.offset = t.offset
  result.data = newSeq[T](t.len)

  let size = t.len * sizeof(T)

  check cudaMemCpy(result.get_data_ptr,
                   t.get_data_ptr,
                   size,
                   cudaMemcpyDeviceToHost)