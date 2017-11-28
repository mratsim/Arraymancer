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

# Nvidia CuDNN backend configuration
# Note: Having CUDA installed does not mean CuDNN is installed

import  nimcuda/cudnn,
        ../../tensor/tensor,
        ../../tensor/backend/cuda
export  cudnn, cuda

# #####################################################################
# CuDNN initialization & release

proc initCudnnHandle(): cudnnHandle_t =
  check cudnnCreate(addr result)

{.experimental.}
proc `=destroy`(c: cudnnHandle_t) =
  check cudnnDestroy(c)

let cudnnHandle0* = initCudnnHandle()

# #####################################################################
# Types and destructors

template asCudnnType*[T: SomeReal](typ: typedesc[T]): cudnnDataType_t =
  when T is float32:
    CUDNN_DATA_FLOAT
  elif T is float64:
    CUDNN_DATA_DOUBLE

{.experimental.}
proc `=destroy`(o: cudnnTensorDescriptor_t) =
  check cudnnDestroyTensorDescriptor o

proc `=destroy`(o: cudnnFilterDescriptor_t) =
  check cudnnDestroyFilterDescriptor o

proc `=destroy`(o: cudnnConvolutionDescriptor_t) =
  check cudnnDestroyConvolutionDescriptor o

# #####################################################################
# Tensor descriptor

proc newCudnn4DTensorDesc*[T: SomeReal](t: CudaTensor[T]): cudnnTensorDescriptor_t {.inline, noinit.}=
  # TODO: destroy descriptor automatically
  # TODO: generalize with the NDTensor Desc
  check cudnnCreateTensorDescriptor addr result

  check cudnnSetTensor4dDescriptorEx(
    result,
    T.asCudnnType,
    t.shape[0].cint, # n
    t.shape[1].cint, # c
    t.shape[2].cint, # h
    t.shape[3].cint, # w
    t.strides[0].cint, # n
    t.strides[1].cint, # c
    t.strides[2].cint, # h
    t.strides[3].cint, # w
  )