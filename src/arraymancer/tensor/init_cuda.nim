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

import  ../private/sequninit,
        ../std_version_types,
        ./private/p_init_cuda,
        ./backend/cuda,
        ./backend/cuda_global_state,
        ./data_structure,
        ./init_cpu

proc cuda*[T:SomeFloat](t: Tensor[T]): CudaTensor[T] {.noinit.}=
  ## Convert a tensor on Cpu to a tensor on a Cuda device.
  # Note: due to modifying the cudaStream0 global var for async copy
  # proc cannot be tagged noSideEffect

  result = newCudaTensor[T](t.shape)

  # TODO: avoid reordering rowMajor tensors. This is only needed for inplace operation in CUBLAS.
  let contig_t = t.asContiguous(colMajor, force = true)
  let size = csize_t(result.size * sizeof(T))

  # For host to device we use non-blocking copy
  # Host can proceed with computation.
  # On CUDA device, next operations will be batch in the stream queue.
  check cudaMemCpyAsync(result.get_data_ptr,
                        contig_t.get_data_ptr,
                        size,
                        cudaMemcpyHostToDevice,
                        cudaStream0) # cudaStream0 is a cudaStream_t global var

proc cpu*[T:SomeFloat](t: CudaTensor[T]): Tensor[T] {.noinit.}=
  ## Convert a tensor on a Cuda device to a tensor on Cpu.
  # We use blocking copy in this case to make sure
  # all data is available for future computation

  result.shape = t.shape
  result.strides = t.strides
  result.offset = t.offset

  allocCpuStorage result.storage, t.storage.Flen

  let size = csize_t(t.storage.Flen * sizeof(T))

  check cudaMemCpy( result.get_data_ptr,
                    t.get_data_ptr,
                    size,
                    cudaMemcpyDeviceToHost)



proc zeros_like*[T: SomeFloat](t: CudaTensor[T]): CudaTensor[T] {.noinit, inline.} =
  ## Creates a new CudaTensor filled with 0 with the same shape as the input
  ## Input:
  ##      - Shape of the CudaTensor
  ##      - Type of its elements
  ## Result:
  ##      - A zero-ed CudaTensor of the same shape

  # TODO use cudaMemset
  result = zeros[T](t.shape).cuda

proc ones_like*[T: SomeFloat](t: CudaTensor[T]): CudaTensor[T] {.noinit, inline.} =
  ## Creates a new CudaTensor filled with 1 with the same shape as the input
  ## and filled with 1
  ## Input:
  ##      - A CudaTensor
  ## Result:
  ##      - A one-ed CudaTensor of the same shape
  result = ones[T](t.shape).cuda
