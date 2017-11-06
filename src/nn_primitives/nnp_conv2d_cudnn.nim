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

type
  SizeHW* = array[2, cint]
    ## height
    ## width

template toPtrCint(s: typed): ptr cint =
  unsafeAddr s[0]

proc check_CuDNN(msg: cudnnStatus_t) =
  echo $msg & " " & $int(msg)

template typeToCudnnType[T: SomeReal]: cudnnDataType_t =
  when T is float32:
    CUDNN_DATA_FLOAT
  elif T is float64:
    CUDNN_DATA_DOUBLE
  else:
    raise newException(ValueError, "Only float32 and float64 are supported")

template newCudnn4DTensorDesc[T: SomeReal](t: CudaTensor[T]): cudnnTensorDescriptor_t =
  # TODO: destroy descriptor automatically
  var td: cudnnTensorDescriptor_t

  echo "\ncudnnCreateTensorDescriptor"
  check_CuDNN cudnnCreateTensorDescriptor addr td

  echo "\ncudnnSetTensor4dDescriptorEx"
  check_CuDNN cudnnSetTensor4dDescriptorEx(
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

  echo "\ncudnnCreateFilterDescriptor"
  check_CuDNN cudnnCreateFilterDescriptor addr fd

  var filters = [out_feats.cint, in_feats.cint, kH.cint, kW.cint]

  echo "\ncudnnSetFilterNdDescriptor"
  check_CuDNN cudnnSetFilterNdDescriptor(
    fd,
    typeToCudnnType[T](),
    CUDNN_TENSOR_NCHW, # TODO do not hardcode the format
    tensorDims,
    addr filters[0]
  )
  fd


proc conv2d*[T: SomeReal](input, filter, bias: CudaTensor[T],
                padA: SizeHW = [0.cint,0],
                filterStrideA: SizeHW = [1.cint,1]): CudaTensor[T] {.inline.} =
  ## Input:
  ##     - ``input`` 4D Tensor batch of images of the size [N,C_in,H_in,W_in]
  ##     - ``filter`` 4D Tensor convolving kernel filters of the size [C_out,C_in,kH,kW]
  ##     - ``bias`` 3D Tensor bias of the size [C_out,1,1] or an empty tensor for no bias

  const convDims: cint = 2
  const tensorDims: cint = 4

  var convDesc: cudnnConvolutionDescriptor_t
  echo "\ncudnnCreateConvolutionDescriptor"
  check_CuDNN cudnnCreateConvolutionDescriptor(addr convDesc)

  echo "Kernel shape"
  echo filter.shape

  let filterDesc = newCudnnFilterDesc[T](
    tensorDims,
    filter.shape[1], # in_features (ex: 3 color channels)
    filter.shape[0], # out_features
    filter.shape[2], # convolving kernel height
    filter.shape[3]  # convolving kernel width
    )
  let srcTensorDesc = newCudnn4DTensorDesc input
  echo input.shape
  echo input.strides

  let upscaleA: SizeHW = [1.cint, 1]

  mixin typeToCudnnType # The template needs mixin to work in an exported proc

  echo "\ncudnnSetConvolutionNdDescriptor"
  check_CuDNN cudnnSetConvolutionNdDescriptor(
    convDesc,
    convDims,
    padA.toPtrCint,
    filterStrideA.toPtrCint,
    upscaleA.toPtrCint,
    CUDNN_CROSS_CORRELATION,
    typeToCudnnType[T]()
  )

  var tensorOutputDimA: array[tensorDims, cint]

  echo "\ncudnnGetConvolutionNdForwardOutputDim"
  check_CuDNN cudnnGetConvolutionNdForwardOutputDim(
    convDesc,
    srcTensorDesc,
    filterDesc,
    tensorDims,
    addr tensorOutputDimA[0]
  )

  ## TODO replace by op in CuDNN dev guide:
  ## Each dimension of the (nbDims-2)-D images of the output tensor is computed as followed:
  ##   outputDim = 1 + ( inputDim + 2*pad - (((filterDim-1)*upscaleA)+1) )/ convolutionStride;

  echo "Manual computation - outputDim = 1 + ( inputDim + 2*pad - (((filterDim-1)*upscaleA)+1) )/ convolutionStride"
  echo "inputDim[2] - height: " & $input.shape[2]
  echo "pad[0] - height: " & $padA[0]
  echo "filterDimA[2] - height: " & $filter.shape[2]
  echo "upscaleA[0] - height: " & $upscaleA[0]
  echo "filterStrideA[0] - height: " & $filterStrideA[0]

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


  ## Result
  result = newCudaTensor[T]([n, c, h, w], colMajor)
  let dstTensorDesc = newCudnn4DTensorDesc result

  var best_algo: cudnnConvolutionFwdAlgo_t

  # TODO make it a parameter so it's only calculated for
  # the very first convolution
  echo "\ncudnnGetConvolutionForwardAlgorithm"
  check_CuDNN cudnnGetConvolutionForwardAlgorithm(
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

  echo "\ncudnnGetConvolutionForwardWorkspaceSize"
  check_CuDNN cudnnGetConvolutionForwardWorkspaceSize(
    defaultHandle_cudnn,
    srcTensorDesc,
    filterDesc,
    convDesc,
    dstTensorDesc,
    best_algo,
    addr sizeInBytes
  )

  var workspace: ptr T = nil

  if sizeInBytes != 0:
    workspace = addr newCudaSeq[T](sizeInBytes div sizeof(T)).data[0] # TODO: newCudaSeq will multiply by sizeof(T) anyway

  var alpha = 1 # scaling factor
  var beta = 0 # scaling factor

  # TODO: use strides

  echo "\nudnnConvolutionForward"
  check_CuDNN cudnnConvolutionForward(
    defaultHandle_cudnn,
    addr alpha,
    srcTensorDesc,
    input.get_offset_ptr,
    filterDesc,
    filter.get_offset_ptr,
    convDesc,
    best_algo,
    workspace,
    sizeInBytes,
    addr beta,
    dstTensorDesc,
    result.get_offset_ptr
  )

  echo result
  echo bias

  result .+= bias.unsafeUnsqueeze(0)

  # TODO: destroy descriptors
