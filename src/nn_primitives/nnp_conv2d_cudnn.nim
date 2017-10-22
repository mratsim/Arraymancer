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
  SizeHW* = array[2, int]
    ## height
    ## width

proc toPtrCint(s: SizeHW): ptr cint =
  var tmp = [s[0].cint, s[1].cint] # TODO: when tmp goes out of scope
  result = addr tmp[0]             # what is happening to the pointer?

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
  rank, in_feats, out_feats, kH, kW: int): cudnnFilterDescriptor_t =
  # TODO: destroy descriptor automatically
  var fd: cudnnFilterDescriptor_t
  check cudnnCreateFilterDescriptor addr fd

  var filters = [in_feats.cint, out_feats.cint, kH.cint, kW.cint]

  check cudnnSetFilterNdDescriptor(
    fd,
    typeToCudnnType[T](),
    CUDNN_TENSOR_NCHW, # TODO do not hardcode the format
    rank.cint,
    addr filters[0]
  )
  fd


proc conv2d*[T: SomeReal](input, weight, bias: CudaTensor[T],
                padding: SizeHW = [0,0],
                stride: SizeHW = [1,1]): CudaTensor[T] {.inline.} =
  ## Input:
  ##     - ``input`` 4D Tensor batch of images of the size [N,C_in,H_in,W_in]
  ##     - ``weight`` 4D Tensor convolving kernel weights of the size [C_out,C_in,kH,kW]
  ##     - ``bias`` 3D Tensor bias of the size [C_out,1,1] or an empty tensor for no bias

  const convDims = 2
  const rank = 4

  var conv_W_cd: cudnnConvolutionDescriptor_t

  let conv_filter_desc = newCudnnFilterDesc[T](
    rank,
    weight.shape[1], # in_features (ex: 3 color channels)
    weight.shape[0], # out_features
    weight.shape[2], # convolving kernel height
    weight.shape[3]  # convolving kernel width
    )
  let conv_in_td = newCudnn4DTensorDesc input

  let dilation = [1, 1]

  mixin typeToCudnnType # The template needs mixin to work in an exported proc

  check cudnnSetConvolutionNdDescriptor(
    conv_W_cd,
    convDims,
    padding.toPtrCint,
    stride.toPtrCint,
    dilation.toPtrCint,
    CUDNN_CROSS_CORRELATION,
    typeToCudnnType[T]()
  )

  var tensorOutputDim: array[rank, cint]

  check cudnnGetConvolutionNdForwardOutputDim(
    conv_W_cd,
    conv_in_td,
    conv_filter_desc,
    rank.cint,
    addr tensorOutputDim[0]
  )

  ## TODO replace by op in CuDNN dev guide:
  ## Each dimension of the (nbDims-2)-D images of the output tensor is computed as followed:
  ##   outputDim = 1 + ( inputDim + 2*pad - (((filterDim-1)*dilation)+1) )/ convolutionStride;

  let
    n = tensorOutputDim[0].int
    c = tensorOutputDim[1].int
    h = tensorOutputDim[2].int
    w = tensorOutputDim[3].int

  result = newCudaTensor[T]([n, c, h, w])

  let conv_out_td = newCudnn4DTensorDesc result

  var best_algo: cudnnConvolutionFwdAlgo_t

  # TODO make it a parameter so it's only calculated for
  # the very first convolution
  check cudnnGetConvolutionForwardAlgorithm(
    defaultHandle_cudnn,
    conv_in_td,
    conv_filter_desc,
    conv_W_cd,
    conv_out_td,
    CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
    0,
    addr best_algo
  )

  # TODO algo comparison with cudnnFindConvolutionForwardAlgorithm

  var sizeInBytes: csize

  check cudnnGetConvolutionForwardWorkspaceSize(
    defaultHandle_cudnn,
    conv_in_td,
    conv_filter_desc,
    conv_W_cd,
    conv_out_td,
    best_algo,
    addr sizeInBytes
  )

  var workspace: ptr T = nil

  if sizeInBytes != 0:
    workspace = addr newCudaSeq[T](sizeInBytes div sizeof(T)).data[0] # TODO: newCudaSeq will multiply by sizeof(T) anyway

  var alpha = 1 # scaling factor
  var beta = 0 # scaling factor

  # TODO: use strides

  check cudnnConvolutionForward(
    defaultHandle_cudnn,
    addr alpha,
    conv_in_td,
    input.get_data_ptr,
    conv_filter_desc,
    weight.get_data_ptr,
    conv_W_cd,
    best_algo,
    workspace,
    sizeInBytes,
    addr beta,
    conv_out_td,
    result.get_data_ptr
  )

  result += bias

  # TODO: destroy descriptors
