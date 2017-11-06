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


import  ../../tensor/backend/cuda,
        ../../tensor/tensor,
        nimcuda/nimcuda,
        ./cudnn

template asCudnnType*[T: SomeReal](typ: typedesc[T]): cudnnDataType_t =
  when T is float32:
    CUDNN_DATA_FLOAT
  elif T is float64:
    CUDNN_DATA_DOUBLE

type
  SizeHW* = array[2, cint]
    ## height
    ## width

template getPtr*(s: SizeHW): ptr cint =
  unsafeAddr s[0]

proc newCudnnConvKernelDesc*[T: SomeReal](
  convKernel: CudaTensor[T]): cudnnFilterDescriptor_t {.inline.}=
  # TODO: destroy descriptor automatically
  check cudnnCreateFilterDescriptor addr result

  var filters = [ convKernel.shape[0].cint, # out features (for example 16 feature maps)
                  convKernel.shape[1].cint, # in features (for example 3 color channels)
                  convKernel.shape[2].cint, # convolving kernel height
                  convKernel.shape[3].cint] # convolving kernel width

  check cudnnSetFilterNdDescriptor(
    result,
    T.asCudnnType,
    CUDNN_TENSOR_NCHW, # TODO do not hardcode the format
    convKernel.rank.cint,
    addr filters[0]
  )

proc newCudnn4DTensorDesc*[T: SomeReal](t: CudaTensor[T]): cudnnTensorDescriptor_t {.inline.}=
  # TODO: destroy descriptor automatically
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


proc convOutDims*(input, kernel: CudaTensor, padding, convStrides, dilation: SizeHW): MetadataArray {.inline.}=

  ## Each dimension of the (nbDims-2)-D images of the output tensor is computed as followed:
  ##   outputDim = 1 + ( inputDim + 2*pad - (((filterDim-1)*upscaleA)+1) )/ convolutionStride;

  ## Input and result are of shape [N, C, H, W]
  ## Kernel of shape [C_out, C_in, h, w]

  let
    iH = input.shape[2]
    iW = input.shape[3]
    pH = padding[0]
    pW = padding[1]
    kH = kernel.shape[2]
    kW = kernel.shape[3]
    dH = dilation[0]
    dW = dilation[1]
    sH = convStrides[0]
    sW = convStrides[1]

  result.len = 4
  result[0] = input.shape[0]
  result[1] = input.shape[1]
  result[2] = 1 + (iH + 2*pH - (((kH-1) * dH) + 1) div sH)
  result[3] = 1 + (iW + 2*pW - (((kW-1) * dW) + 1) div sW)

  # # Cudnn version
  # var tensorOutputDimA: array[4, cint] # 2D convolution -> NCHW order
  # check cudnnGetConvolutionNdForwardOutputDim(
  #   convDesc,
  #   srcTensorDesc,
  #   filterDesc,
  #   input.rank.cint,
  #   addr tensorOutputDimA[0]
  # )

type ConvAlgoSpace*[T] = object
  algo*: cudnnConvolutionFwdAlgo_t # TODO: create a deallocCudnn so that descriptor are destroyed by Nim GC.
  workspace*: ref[ptr T]
  sizeInBytes*: csize

proc conv_algo_workspace*[T: SomeReal](
  srcTensorDesc: cudnnTensorDescriptor_t,
  convKernelDesc: cudnnFilterDescriptor_t,
  convDesc: cudnnConvolutionDescriptor_t,
  dstTensorDesc: cudnnTensorDescriptor_t
): ConvAlgoSpace[T] =

  check cudnnGetConvolutionForwardAlgorithm(
    defaultHandle_cudnn,
    srcTensorDesc,
    convKernelDesc,
    convDesc,
    dstTensorDesc,
    CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
    0,
    addr result.algo
  )

  when defined(debug):
    echo "\nCudnn Algorithm selected: " & $best_algo

  check cudnnGetConvolutionForwardWorkspaceSize(
    defaultHandle_cudnn,
    srcTensorDesc,
    convKernelDesc,
    convDesc,
    dstTensorDesc,
    result.algo,
    addr result.sizeInBytes
  )

  when defined(debug):
    echo "\nCudnn convolution 2D - Workspace size: " & $sizeInBytes & " bytes"

  if result.sizeInBytes != 0:
    new(result.workspace, deallocCuda)
    # cudaMalloc multiply by sizeof(T) so we must divide before hand
    result.workspace[] = cudaMalloc[T](result.sizeInBytes div sizeof(T))