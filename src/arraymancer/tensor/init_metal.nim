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
      ./private/p_init_metal,
      ./backend/metal/metal_backend,
      ./data_structure,
      ./init_cpu

proc metal*[T: SomeFloat](t: Tensor[T]): MetalTensor[T] {.noinit.} =
  ## Convert a tensor on CPU to a tensor on a Metal device.
  ## This performs a synchronous copy to the GPU.

  result = newMetalTensor[T](t.shape)

  let contig_t = t.asContiguous(colMajor, force = true)
  let size = result.size * sizeof(T)

  uploadToBuffer(result.storage.Fbuffer, contig_t.get_data_ptr, size)

  # Store a CPU copy for lazy transfer (optional optimization)
  # For now, we leave it empty to avoid unnecessary memory duplication
  result.cpuData = @[]

proc cpu*[T: SomeFloat](t: MetalTensor[T]): Tensor[T] {.noinit.} =
  ## Convert a tensor on a Metal device to a tensor on CPU.
  ## This performs a synchronous copy from the GPU.

  result.shape = t.shape
  result.strides = t.strides
  result.offset = t.offset

  allocCpuStorage result.storage, t.storage.Flen

  let size = t.storage.Flen * sizeof(T)
  downloadFromBuffer(t.storage.Fbuffer, result.get_data_ptr, size)

proc toMetal*[T: SomeFloat](t: Tensor[T]): MetalTensor[T] {.noinit.} =
  ## Convert a tensor on CPU to a tensor on a Metal device (lazy transfer).
  ## The data is copied immediately for now; future versions may defer transfer.
  metal(t)

proc toCpu*[T: SomeFloat](t: MetalTensor[T]): Tensor[T] {.noinit.} =
  ## Convert a tensor on a Metal device to a tensor on CPU (lazy transfer).
  ## The data is copied immediately for now; future versions may defer transfer.
  cpu(t)

proc zeros_like*[T: SomeFloat](t: MetalTensor[T]): MetalTensor[T] {.noinit, inline.} =
  ## Creates a new MetalTensor filled with 0 with the same shape as the input
  result = zeros[T](t.shape).metal

proc ones_like*[T: SomeFloat](t: MetalTensor[T]): MetalTensor[T] {.noinit, inline.} =
  ## Creates a new MetalTensor filled with 1 with the same shape as the input
  result = ones[T](t.shape).metal
