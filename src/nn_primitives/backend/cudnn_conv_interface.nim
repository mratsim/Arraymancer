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
        ./cudnn

# #####################################################################
# Base types

type
  SizeHW* = array[2, int] # Todo, unify with NNPACK Size2D
    ## height
    ## width

  ConvConfig*[N: static[int]] = object
    pad*: array[N, cint]
    strides*: array[N, cint]
    dilation*: array[N, cint]

type Algo = cudnnConvolutionFwdAlgo_t or cudnnConvolutionBwdFilterAlgo_t or cudnnConvolutionBwdDataAlgo_t

type ConvAlgoSpace*[T: SomeFloat, Algo] = object
  algo*: Algo
  workspace*: ref[ptr T]
  sizeInBytes*: csize

# #####################################################################
# Convolution descriptor

proc initConv2DConfig(padding, strides, dilation: SizeHW): ConvConfig[2] {.inline, noInit.}=
  result.pad = [padding[0].cint, padding[1].cint]
  result.strides = [strides[0].cint, strides[1].cint]
  result.dilation = [dilation[0].cint, dilation[1].cint]

template getPtr[N: static[int]](convConfig: ConvConfig[N], field: untyped): ptr cint =
  unsafeAddr convConfig.field[0]

proc newConvDesc( convolution_dimension: range[2..3],
                  convolution_config: ConvConfig,
                  convolution_mode: cudnnConvolutionMode_t,
                  T: typedesc): cudnnConvolutionDescriptor_t {.inline, noInit.}=
  check cudnnCreateConvolutionDescriptor(addr result)
  check cudnnSetConvolutionNdDescriptor(
    result,
    convolution_dimension.cint,
    convolution_config.getPtr(pad),
    convolution_config.getPtr(strides),
    convolution_config.getPtr(dilation),
    CUDNN_CROSS_CORRELATION,
    T.asCudnnType
  )

proc newConv2dDesc*[T: SomeFloat](padding, strides, dilation: SizeHW): cudnnConvolutionDescriptor_t {.noInit, inline.}=
  let convConfig = initConv2DConfig(padding, strides, dilation)
  result = newConvDesc(2, convConfig, CUDNN_CROSS_CORRELATION, T)

# #####################################################################
# Convolution kernel descriptor

proc newCudnnConvKernelDesc*[T: SomeFloat](
  convKernel: CudaTensor[T]): cudnnFilterDescriptor_t {.inline, noInit.}=
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

proc convOutDims*(input, kernel: CudaTensor, padding, strides, dilation: SizeHW): MetadataArray {.inline, noInit.}=

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
    sH = strides[0]
    sW = strides[1]

  result.len = 4
  result[0] = input.shape[0]
  result[1] = kernel.shape[0]
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


# ###############################################################
# Forward convolution: Algorithm and Worksize space

proc conv_algo_workspace*[T: SomeFloat](
  srcTensorDesc: cudnnTensorDescriptor_t,
  kernelDesc: cudnnFilterDescriptor_t,
  convDesc: cudnnConvolutionDescriptor_t,
  dstTensorDesc: cudnnTensorDescriptor_t
): ConvAlgoSpace[T, cudnnConvolutionFwdAlgo_t] {.noInit.}=

  when defined(debug):
    echo "\nCudnn conv2d - get forward algorithm"

  check cudnnGetConvolutionForwardAlgorithm(
    cudnnHandle0,
    srcTensorDesc,
    kernelDesc,
    convDesc,
    dstTensorDesc,
    CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
    0,
    addr result.algo
  )

  when defined(debug):
    echo "Cudnn conv2d - forward algorithm selected: " & $result.algo

  check cudnnGetConvolutionForwardWorkspaceSize(
    cudnnHandle0,
    srcTensorDesc,
    kernelDesc,
    convDesc,
    dstTensorDesc,
    result.algo,
    addr result.sizeInBytes
  )

  when defined(debug):
    echo "Cudnn conv2D - workspace size: " & $result.sizeInBytes & " bytes"

  if result.sizeInBytes != 0:
    new(result.workspace, deallocCuda)
    # cudaMalloc multiply by sizeof(T) so we must divide before hand
    result.workspace[] = cudaMalloc[T](result.sizeInBytes div sizeof(T))

# ###############################################################
# Backward convolution - Kernel: Algorithm and Worksize space

proc conv_bwd_kernel_algo_workspace*[T: SomeFloat](
  srcTensorDesc: cudnnTensorDescriptor_t,
  gradOutputTensorDesc: cudnnTensorDescriptor_t, # gradOuput is the gradient of the output. Result will be the gradient of the input
  gradKernelDesc: cudnnFilterDescriptor_t,
  convDesc: cudnnConvolutionDescriptor_t
): ConvAlgoSpace[T, cudnnConvolutionBwdFilterAlgo_t] {.noInit.}=

  when defined(debug):
    echo "\nCudnn conv2d - get backward kernel algorithm"

  check cudnnGetConvolutionBackwardFilterAlgorithm(
    cudnnHandle0,
    srcTensorDesc,
    gradOutputTensorDesc,
    convDesc,
    gradKernelDesc,
    CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST,
    0,
    addr result.algo
  )

  when defined(debug):
    echo "Cudnn conv2d - backward kernel algorithm selected: " & $result.algo

  check cudnnGetConvolutionBackwardFilterWorkspaceSize(
    cudnnHandle0,
    srcTensorDesc,
    gradOutputTensorDesc,
    convDesc,
    gradKernelDesc,
    result.algo,
    addr result.sizeInBytes
  )

  when defined(debug):
    echo "Cudnn conv2d - backward kernel workspace size: " & $result.sizeInBytes & " bytes"

  if result.sizeInBytes != 0:
    new(result.workspace, deallocCuda)
    # cudaMalloc multiply by sizeof(T) so we must divide before hand
    result.workspace[] = cudaMalloc[T](result.sizeInBytes div sizeof(T))

# ###############################################################
# Backward convolution - Data: Algorithm and Worksize space

proc conv_bwd_data_algo_workspace*[T: SomeFloat](
  srcTensorDesc: cudnnTensorDescriptor_t,
  gradOutputTensorDesc: cudnnTensorDescriptor_t, # gradOuput is the gradient of the output. Result will be the gradient of the input
  kernelDesc: cudnnFilterDescriptor_t,
  convDesc: cudnnConvolutionDescriptor_t,
  gradInputTensorDesc: cudnnTensorDescriptor_t # gradInput is what we want to compute in the backward pass
): ConvAlgoSpace[T, cudnnConvolutionBwdDataAlgo_t] {.noInit.}=

  when defined(debug):
    echo "\nCudnn conv2d - get backward data algorithm"

  check cudnnGetConvolutionBackwardDataAlgorithm(
    cudnnHandle0,
    kernelDesc,
    gradOutputTensorDesc,
    convDesc,
    gradInputTensorDesc,
    CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST,
    0,
    addr result.algo
  )

  when defined(debug):
    echo "Cudnn conv2d - backward data algorithm selected: " & $result.algo

  check cudnnGetConvolutionBackwardDataWorkspaceSize(
    cudnnHandle0,
    kernelDesc,
    gradOutputTensorDesc,
    convDesc,
    gradInputTensorDesc,
    result.algo,
    addr result.sizeInBytes
  )

  when defined(debug):
    echo "Cudnn conv2d - backward data workspace size: " & $result.sizeInBytes & " bytes"

  if result.sizeInBytes != 0:
    new(result.workspace, deallocCuda)
    # cudaMalloc multiply by sizeof(T) so we must divide before hand
    result.workspace[] = cudaMalloc[T](result.sizeInBytes div sizeof(T))
