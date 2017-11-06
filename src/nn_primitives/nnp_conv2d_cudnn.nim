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

import  ./backend/cudnn,
        ../tensor/tensor,
        ../tensor/backend/cuda,
        ../tensor/private/p_init_cuda # TODO: it might be worth it to export newCudaTensor

import nimcuda/[cuda_runtime_api, nimcuda]

type
  SizeHW* = array[2, cint]
    ## height
    ## width

template toPtrCint(s: typed): ptr cint =
  unsafeAddr s[0]

proc typeToCudnnType[T: SomeReal]: cudnnDataType_t =
  when T is float32:
    CUDNN_DATA_FLOAT
  elif T is float64:
    CUDNN_DATA_DOUBLE
  else:
    raise newException(ValueError, "Only float32 and float64 are supported")

proc newCudnn4DTensorDesc[T: SomeReal](t: CudaTensor[T]): cudnnTensorDescriptor_t =
  # TODO: destroy descriptor automatically
  var td: cudnnTensorDescriptor_t

  check cudnnCreateTensorDescriptor addr td

  check cudnnSetTensor4dDescriptorEx(
    td,
    typeToCudnnType[T](),
    t.shape[0].cint, # n
    t.shape[1].cint, # c
    t.shape[2].cint, # h
    t.shape[3].cint, # w
    t.strides[0].cint, # n
    t.strides[1].cint, # c
    t.strides[2].cint, # h
    t.strides[3].cint, # w
  )
  td

template newCudnnFilterDesc[T: SomeReal](
  tensorDims: cint, in_feats, out_feats, kH, kW: int): cudnnFilterDescriptor_t =
  # TODO: destroy descriptor automatically
  var fd: cudnnFilterDescriptor_t

  check cudnnCreateFilterDescriptor addr fd

  var filters = [out_feats.cint, in_feats.cint, kH.cint, kW.cint]

  check cudnnSetFilterNdDescriptor(
    fd,
    typeToCudnnType[T](),
    CUDNN_TENSOR_NCHW, # TODO do not hardcode the format
    tensorDims,
    addr filters[0]
  )
  fd

proc conv2d*[T: SomeReal](input, filter, bias: CudaTensor[T],
                padA: SizeHW = [0.cint,0],
                filterStrideA: SizeHW = [1.cint,1]): CudaTensor[T] =
  ## Input:
  ##     - ``input`` 4D Tensor batch of images of the size [N,C_in,H_in,W_in]
  ##     - ``filter`` 4D Tensor convolving kernel filters of the size [C_out,C_in,kH,kW]
  ##     - ``bias`` 3D Tensor bias of the size [C_out,1,1] or an empty tensor for no bias

  const convDims: cint = 2
  const tensorDims: cint = 4

  var convDesc: cudnnConvolutionDescriptor_t
  check cudnnCreateConvolutionDescriptor(addr convDesc)


  let filterDesc = newCudnnFilterDesc[T](
    tensorDims,
    filter.shape[1], # in_features (ex: 3 color channels)
    filter.shape[0], # out_features
    filter.shape[2], # convolving kernel height
    filter.shape[3]  # convolving kernel width
    )
  let srcTensorDesc = newCudnn4DTensorDesc input

  let upscaleA: SizeHW = [1.cint, 1]

  mixin typeToCudnnType # The template needs mixin to work in an exported proc

  check cudnnSetConvolutionNdDescriptor(
    convDesc,
    convDims,
    padA.toPtrCint,
    filterStrideA.toPtrCint,
    upscaleA.toPtrCint,
    CUDNN_CROSS_CORRELATION,
    typeToCudnnType[T]()
  )

  var tensorOutputDimA: array[tensorDims, cint]

  check cudnnGetConvolutionNdForwardOutputDim(
    convDesc,
    srcTensorDesc,
    filterDesc,
    tensorDims,
    addr tensorOutputDimA[0]
  )

  ## TODO replace by op in CuDNN dev guide:
  ## Each dimension of the (nbDims-2)-D images of the output tensor is computed as followed:
  ##   outputDim = 1 + ( inputDim + 2*pad - (((filterDim-1)*upscaleA)+1) )/ convolutionStride;

  echo "Manual Computation: " & $(
    1 + (input.shape[2] + 2*padA[0] -
          (
            (
              (filter.shape[2]-1) * upscaleA[0]
            ) + 1
          ) div filterStrideA[0]
        )
    )

  let
    n = tensorOutputDimA[0].int
    c = tensorOutputDimA[1].int
    h = tensorOutputDimA[2].int
    w = tensorOutputDimA[3].int

  echo "Computed output dims: " & $[n, c, h, w]

  result = newCudaTensor[T]([n, c, h, w], rowMajor)
  let dstTensorDesc = newCudnn4DTensorDesc result

  var best_algo: cudnnConvolutionFwdAlgo_t

  # TODO make it a parameter so it's only calculated for
  # the very first convolution
  check cudnnGetConvolutionForwardAlgorithm(
    defaultHandle_cudnn,
    srcTensorDesc,
    filterDesc,
    convDesc,
    dstTensorDesc,
    CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
    0,
    addr best_algo
  )

  # TODO algo comparison with cudnnFindConvolutionForwardAlgorithm
  echo "\nBest algo: " & $best_algo

  var sizeInBytes: csize

  check cudnnGetConvolutionForwardWorkspaceSize(
    defaultHandle_cudnn,
    srcTensorDesc,
    filterDesc,
    convDesc,
    dstTensorDesc,
    best_algo,
    addr sizeInBytes
  )

  echo "\nWorkspace size: " & $sizeInBytes & " bytes"
  var workspace: ptr T = nil

  if sizeInBytes != 0:
    workspace = addr newCudaSeq[T](sizeInBytes div sizeof(T)).data[0] # TODO: newCudaSeq will multiply by sizeof(T) anyway
    # TODO garbage collection

  var alpha:T = 1 # scaling factor
  var beta:T = 0 # scaling factor

  check cudnnConvolutionForward(
    defaultHandle_cudnn,
    addr alpha,
    srcTensorDesc,
    input.data.data[],
    filterDesc,
    filter.data.data[],
    convDesc,
    best_algo,
    workspace,
    sizeInBytes,
    addr beta,
    dstTensorDesc,
    result.data.data[]
  )

  result .+= bias.unsafeUnsqueeze(0)

  # TODO: destroy descriptors