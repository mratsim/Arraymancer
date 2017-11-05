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

import  ../data_structure,
        ./global_config,
        nimcuda/[nimcuda, cuda_runtime_api, driver_types]

export nimcuda.check

# Data structures to ease interfacing with Cuda and kernels

proc cudaMalloc[T](size: Natural): ptr T {.noSideEffect, inline.}=
  ## Internal proc.
  ## Wrap CudaMAlloc(var pointer, size) -> Error_code
  let s = size * sizeof(T)
  check cudaMalloc(cast[ptr pointer](addr result), s)

proc deallocCuda[T](p: ref[ptr T]) {.noSideEffect.}=
  if not p[].isNil:
    check cudaFree(p[])


# ##############################################################
# # Base CudaSeq type
# # End goal is for it to have value semantics like Nim seq

proc newCudaSeq*[T: SomeReal](length: Natural): CudaSeq[T] {.noSideEffect.}=
  result.len = length
  new(result.data, deallocCuda)
  result.data[] = cast[ptr UncheckedArray[T]](cudaMalloc[T](result.len))

# #########################################################
# # Sending tensor layout to Cuda Kernel

# # So that layout->strides can be used in Cuda kernel, it's easier if everything is declared from cpp
# # pending https://github.com/nim-lang/Nim/issues/6415
#
# template create_CudaTensorLayout(N: static[int]) =
#   ## This Layout in C++ will be overriden by a CudaMemCpy from the Nim data structure
#   {. emit:[ """
#
#     template <typename T>
#     struct CudaTensorLayout {
#       int rank;
#       int shape[""", N,"""];
#       int strides[""", N,"""];
#       int offset;
#       T * __restrict__ data;
#       };
#
#
#   """].}
#
# create_CudaTensorLayout(MAXRANK)

type
  # CudaLayoutArray = array[MAXRANK, cint]
  # This will replace the current ref[ptr T] for shape and strides in the future
  ## Using arrays instead of seq avoids having to indicate __restrict__ everywhere to indicate no-aliasing
  ## We also prefer stack allocated array sice the data will be used at every single loop iteration to compute elements position.
  ## Ultimately it avoids worrying about deallocation too
  CudaLayoutArray = ref[ptr cint]


  CudaTensorLayout [T: SomeReal] = object
    ## Mimicks CudaTensor
    ## This will be stored on GPU in the end
    ## Goal is to avoids clumbering proc with cudaMemcpyshape, strides, offset, data, rank, len
    ##
    ## Check https://github.com/mratsim/Arraymancer/issues/26 (Optimizing Host <-> Cuda transfer)
    ## on why I don't (yet?) use Unified Memory and choose to manage it manually.

    rank*: cint               # Number of dimension of the tensor
    shape*: CudaLayoutArray
    strides*: CudaLayoutArray
    offset*: cint
    data*: ptr T              # Data on Cuda device
    len*: cint                # Number of elements allocated in memory

proc layoutOnDevice*[T:SomeReal](t: CudaTensor[T]): CudaTensorLayout[T] {.noSideEffect.}=
  ## Store a CudaTensor shape, strides, etc information on the GPU
  #
  # TODO: instead of storing pointers to shape/stride/etc that are passed to each kernel
  # pass the layout object directly and call it with layout->shape, layout->rank

  result.rank = t.rank.cint

  result.offset = t.offset.cint
  result.data = t.get_data_ptr
  result.len = t.size.cint

  new result.shape, deallocCuda
  new result.strides, deallocCuda

  result.shape[] = cudaMalloc[cint](MAXRANK)
  result.strides[] = cudaMalloc[cint](MAXRANK)

  var
    tmp_shape: array[MAXRANK, cint] # CudaLayoutArray
    tmp_strides: array[MAXRANK, cint] # CudaLayoutArray

  for i in 0..<result.rank:
    tmp_shape[i] = t.shape[i].cint
    tmp_strides[i] = t.strides[i].cint


  # TODO: use streams and async
  let size = t.rank * sizeof(cint)
  check cudaMemCpy(result.shape[], addr tmp_shape[0], size, cudaMemcpyHostToDevice)
  check cudaMemCpy(result.strides[], addr tmp_strides[0], size, cudaMemcpyHostToDevice)