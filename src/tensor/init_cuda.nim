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

import  ../private/sequninit,
        ./private/p_init_cuda,
        ./backend/cuda,
        ./backend/cuda_global_state,
        ./backend/metadataArray,
        ./data_structure,
        ./init_cpu,
        nimcuda/[cuda_runtime_api, driver_types]

proc clone*[T](t: CudaTensor[T]): CudaTensor[T] {.noInit.}=
  ## Clone (deep copy) a CudaTensor.
  ## Copy will not share its data with the original.
  ##
  ## Tensor is copied as is. For example it will not be made contiguous.
  ## Use `asContiguous` for this case

  # Note: due to modifying the defaultStream global var for async memcopy
  # proc cannot be tagged noSideEffect

  result.shape = t.shape
  result.strides = t.strides
  result.offset = t.offset
  result.storage = newCudaStorage[T](t.storage.Flen)
  let size = t.storage.Flen * sizeof(T)

  check cudaMemCpyAsync(result.get_data_ptr,
                        t.get_data_ptr,
                        size,
                        cudaMemcpyDeviceToDevice,
                        defaultStream) # defaultStream is a cudaStream_t global var

proc cuda*[T:SomeReal](t: Tensor[T]): CudaTensor[T] {.noInit.}=
  ## Convert a tensor on Cpu to a tensor on a Cuda device.
  # Note: due to modifying the defaultStream global var for async copy
  # proc cannot be tagged noSideEffect

  result = newCudaTensor[T](t.shape)

  # TODO: avoid reordering rowMajor tensors. This is only needed for inplace operation in CUBLAS.
  let contig_t = t.asContiguous(colMajor, force = true)
  let size = csize(result.size * sizeof(T))

  # For host to device we use non-blocking copy
  # Host can proceed with computation.
  # On CUDA device, next operations will be batch in the stream queue.
  check cudaMemCpyAsync(result.get_data_ptr,
                        contig_t.get_data_ptr,
                        size,
                        cudaMemcpyHostToDevice,
                        defaultStream) # defaultStream is a cudaStream_t global var

proc cpu*[T:SomeReal](t: CudaTensor[T]): Tensor[T] {.noSideEffect, noInit.}=
  ## Convert a tensor on a Cuda device to a tensor on Cpu.
  # We use blocking copy in this case to make sure
  # all data is available for future computation

  result.shape = t.shape
  result.strides = t.strides
  result.offset = t.offset
  result.data = newSeqUninit[T](t.storage.Flen) # We copy over all the memory allocated

  let size = csize(t.storage.Flen * sizeof(T))

  check cudaMemCpy( result.get_data_ptr,
                    t.get_data_ptr,
                    size,
                    cudaMemcpyDeviceToHost)



proc zeros_like*[T: SomeReal](t: CudaTensor[T]): CudaTensor[T] {.noInit, inline.} =
  ## Creates a new CudaTensor filled with 0 with the same shape as the input
  ## Input:
  ##      - Shape of the CudaTensor
  ##      - Type of its elements
  ## Result:
  ##      - A zero-ed CudaTensor of the same shape

  # TODO use cudaMemset
  result = zeros[T](t.shape).cuda

proc ones_like*[T: SomeReal](t: CudaTensor[T]): CudaTensor[T] {.noInit, inline.} =
  ## Creates a new CudaTensor filled with 1 with the same shape as the input
  ## and filled with 1
  ## Input:
  ##      - A CudaTensor
  ## Result:
  ##      - A one-ed CudaTensor of the same shape
  result = ones[T](t.shape).cuda