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
        ./data_structure,
        nimcuda/[cuda_runtime_api, driver_types]

proc unsafeView*[T](t: CudaTensor[T]): CudaTensor[T] {.inline,noSideEffect.}=
  ## Input:
  ##     - A CudaTensor
  ## Returns:
  ##     - A shallow copy.
  ##
  ## Warning ⚠
  ##   Both tensors shares the same memory. Data modification on one will be reflected on the other.
  ##   However modifying the shape, strides or offset will not affect the other.

  # shape and strides fields have value semantics by default
  # CudaSeq has ref semantics
  system.`=`(result, t)

proc clone*[T](t: CudaTensor[T]): CudaTensor[T] {.noInit.}=
  ## Clone (deep copy) a CudaTensor.
  ## Copy will not share its data with the original.
  ##
  ## Tensor is copied as is. For example it will not be made contiguous.
  ## Use `unsafeContiguous` for this case

  # Note: due to modifying the defaultStream global var for async memcopy
  # proc cannot be tagged noSideEffect

  result.shape = t.shape
  result.strides = t.strides
  result.offset = t.offset
  result.data = newCudaSeq[T](t.data.len)
  let size = t.data.len * sizeof(T)

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
#   dest.shape = src.shape
#   dest.strides = src.strides
#   dest.offset = src.offset
#   dest.data = newCudaSeq(src.data.len)
#
#   let size = dest.size * sizeof(T)
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

proc cuda*[T:SomeReal](t: Tensor[T]): CudaTensor[T] {.noInit.}=
  ## Convert a tensor on Cpu to a tensor on a Cuda device.
  # Note: due to modifying the defaultStream global var for async copy
  # proc cannot be tagged noSideEffect

  result = newCudaTensor[T](t.shape)

  # TODO: avoid reordering rowMajor tensors. This is only needed for inplace operation in CUBLAS.
  let contig_t = t.unsafeContiguous(colMajor, force = true)
  let size = result.size * sizeof(T)

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
  result.data = newSeqUninit[T](t.data.len) # We copy over all the memory allocated

  let size = t.data.len * sizeof(T)

  check cudaMemCpy(result.get_data_ptr,
                   t.get_data_ptr,
                   size,
                   cudaMemcpyDeviceToHost)