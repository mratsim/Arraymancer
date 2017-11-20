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
        ./backend/cudnn_interface,
        ../tensor/tensor,
        ../tensor/backend/cuda,
        ../tensor/private/p_init_cuda # TODO: it might be worth it to export newCudaTensor

import nimcuda/[cuda_runtime_api, nimcuda]

proc conv2d*[T: SomeReal](input, kernel, bias: CudaTensor[T],
                padding: SizeHW = [0,0],
                convStrides, dilation: SizeHW = [1,1]): CudaTensor[T] {.noInit.}=
  ## Input:
  ##     - ``input`` 4D Tensor batch of images of the size [N,C_in,H_in,W_in]
  ##     - ``kernel`` 4D Tensor convolving kernel filters of the size [C_out,C_in,kH,kW]
  ##     - ``bias`` 3D Tensor bias of the size [C_out,1,1]

  # TODO bias as an Optional type

  const convDims: cint = 2
  const rank: cint = 4
  let srcTensorDesc = newCudnn4DTensorDesc input # TODO: destructor
  let kernelDesc = newCudnnConvKernelDesc(kernel) # TODO: destructor

  # Conversion to cint + object living long enough so we can use pointers to it for CuDNN
  let convConfig = initConv2DConfig(padding, convStrides, dilation)

  var convDesc: cudnnConvolutionDescriptor_t # TODO: destructor
  check cudnnCreateConvolutionDescriptor(addr convDesc)

  check cudnnSetConvolutionNdDescriptor(
    convDesc,
    convDims,
    convConfig.getPtr(pad),
    convConfig.getPtr(strides),
    convConfig.getPtr(dilation),
    CUDNN_CROSS_CORRELATION,
    T.asCudnnType
  )

  # Setting up the result
  let result_shape = convOutDims(input, kernel, padding, convStrides, dilation)
  result = newCudaTensor[T](result_shape, rowMajor)
  let dstTensorDesc = newCudnn4DTensorDesc result

  # Getting the convolution algorithm, shared memory workspace and its size.
  let algo_workspace = conv_algo_workspace[T](srcTensorDesc, kernelDesc, convDesc, dstTensorDesc)

  # Scaling factors
  var alpha: T = 1
  var beta: T = 0

  check cudnnConvolutionForward(
    defaultHandle_cudnn,
    addr alpha,
    srcTensorDesc,
    input.data.data[],
    kernelDesc,
    kernel.data.data[],
    convDesc,
    algo_workspace.algo,
    algo_workspace.workspace[],
    algo_workspace.sizeInBytes,
    addr beta,
    dstTensorDesc,
    result.data.data[]
  )

  result .+= bias.unsafeUnsqueeze(0)

proc conv2d_backward*[T](input, kernel, bias: CudaTensor[T],
                         padding: SizeHW = [0,0],
                         convStrides, dilation: SizeHW = [1,1],
                         grad_output: CudaTensor[T],
                         grad_input, grad_kernel, grad_bias: var CudaTensor[T]) =
  ## Computes gradients of a 2D convolution. Intended to be used after
  ## ``conv2d`` to calculate gradients in backward pass.
  ##
  ## Input:
  ##     - ``input`` 4D Tensor batch of images of the size [N,C_in,H_in,W_in]
  ##     - ``kernel`` 4D Tensor convolving kernel weights of the size [C_out,C_in,kH,kW]
  ##     - ``bias`` 3D Tensor bias of the size [C_out,1,1] or an empty tensor for no bias
  ##     - ``padding`` SizeHW tuple with height and width of the padding
  ##     - ``convStrides`` SizeHW tuple with height and width of the convolution strides
  ##     - ``dilation`` SizeHW tuple with a rescaling factor of the convolution
  ##     - ``grad_output`` 4D tensor gradient of the next layer of the size [N,C_out,H_out,W_out]
  ##     - ``grad_input`` tensor where the gradient w.r.t input will be written
  ##     - ``grad_kernel`` tensor where the gradient w.r.t convolution kernel will be written
  ##     - ``grad_bias`` tensor where the gradient w.r.t bias will be written
  ##
  ## Note:
  ##   ``grad_input``, ``grad_kernel`` and ``grad_bias`` will be overwritten. They must have the same shape
  ##    as the corresponding ``input``, ``kernel`` and ``bias``


  const convDims: cint = 2
  const rank: cint = 4



  let # TODO: Automatic destructor
    srcTensorDesc = newCudnn4DTensorDesc input
    kernelDesc = newCudnnConvKernelDesc(kernel)
    gradKernelDesc = newCudnnConvKernelDesc(grad_kernel)
    gradInputTensorDesc =  newCudnn4DTensorDesc grad_input
    gradOutputTensorDesc =  newCudnn4DTensorDesc grad_output

  # Conversion to cint + object living long enough so we can use pointers to it for CuDNN
  let convConfig = initConv2DConfig(padding, convStrides, dilation)

  var convDesc: cudnnConvolutionDescriptor_t # TODO: destructor
  check cudnnCreateConvolutionDescriptor(addr convDesc)

  check cudnnSetConvolutionNdDescriptor(
    convDesc,
    convDims,
    convConfig.getPtr(pad),
    convConfig.getPtr(strides),
    convConfig.getPtr(dilation),
    CUDNN_CROSS_CORRELATION,
    T.asCudnnType
  )

  # Scaling factors
  var alpha: T = 1
  var beta: T = 0

  # Input: getting the backward conv algorithm, shared memory workspace and its size.
  let gradInput_algo_workspace = conv_bwd_data_algo_workspace[T](
                                srcTensorDesc,
                                gradOutputTensorDesc,
                                kernelDesc, # Note the kernel
                                convDesc,
                                gradInputTensorDesc
                                )

  # Input gradient
  check cudnnConvolutionBackwardData(
    defaultHandle_cudnn,
    addr alpha,
    kernelDesc,
    kernel.data.data[],
    gradOutputTensorDesc,
    grad_output.data.data[],
    convDesc,
    gradInput_algo_workspace.algo,
    gradInput_algo_workspace.workspace[],
    gradInput_algo_workspace.sizeInBytes,
    addr beta,
    gradInputTensorDesc,
    grad_input.data.data[]
  )

  # Kernel: getting the backward conv algorithm, shared memory workspace and its size.
  let kernel_algo_workspace = conv_bwd_kernel_algo_workspace[T](
                                srcTensorDesc,
                                gradOutputTensorDesc,
                                gradKernelDesc, # Note the gradKernel
                                convDesc
                                )


  # Kernel gradient
  check cudnnConvolutionBackwardFilter(
    defaultHandle_cudnn,
    addr alpha,
    srcTensorDesc,
    input.data.data[],
    gradOutputTensorDesc,
    grad_output.data.data[],
    convDesc,
    kernel_algo_workspace.algo,
    kernel_algo_workspace.workspace[],
    kernel_algo_workspace.sizeInBytes,
    addr beta,
    gradKernelDesc,
    grad_kernel.data.data[]
  )

  # Bias gradient
  if bias.rank > 0:
    let gradBiasTensorDesc = newCudnn4DTensorDesc grad_bias.unsafeUnsqueeze(0)
    check cudnnConvolutionBackwardBias(
      defaultHandle_cudnn,
      addr alpha,
      gradOutputTensorDesc,
      grad_output.data.data[],
      addr beta,
      gradBiasTensorDesc,
      grad_bias.data.data[]
    )
