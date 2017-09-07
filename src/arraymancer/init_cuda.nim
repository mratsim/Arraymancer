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

proc newCudaTensor[T: SomeReal](shape: openarray[int]): CudaTensor[T] {.noSideEffect.}=
  ## Internal proc
  ## Allocate a CudaTensor
  ## WARNING: The Cuda memory is not initialized to 0
  new(result.data_ref, deallocCuda)
  result.shape = @shape
  result.len = result.shape.product
  result.data_ref[] = cudaMalloc[T](result.len)
  result.strides = shape_to_strides(result.shape)
  result.offset = 0

proc cuda*[T:SomeReal](t: Tensor[T]): CudaTensor[T] {.noSideEffect.}=
  ## Convert a tensor on Cpu to a tensor on a Cuda device.
  result = newCudaTensor[T](t.shape)

  let contig_t = t.asContiguous()
  let size = result.len * sizeof(T)

  check cudaMemCpy(result.get_data_ptr,
                   contig_t.get_data_ptr,
                   size,
                   cudaMemcpyHostToDevice)

proc cpu*[T:SomeReal](t: CudaTensor[T]): Tensor[T] {.noSideEffect.}=
  ## Convert a tensor on a Cuda device to a tensor on Cpu.

  result.shape = t.shape
  result.strides = t.strides
  result.offset = t.offset
  result.data = newSeq[T](t.len)

  let size = t.len * sizeof(T)

  check cudaMemCpy(result.get_data_ptr,
                   t.get_data_ptr,
                   size,
                   cudaMemcpyDeviceToHost)