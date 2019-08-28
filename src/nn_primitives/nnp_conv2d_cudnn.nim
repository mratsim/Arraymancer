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
        ./backend/cudnn_conv_interface,
        ../tensor/tensor,
        ../tensor/private/p_init_cuda # TODO: it might be worth it to export newCudaTensor

proc conv2d*[T: SomeFloat](input, kernel, bias: CudaTensor[T],
                padding: SizeHW = [0,0],
                strides, dilation: SizeHW = [1,1]): CudaTensor[T] {.noInit.}=
  ## Input:
  ##     - ``input`` 4D Tensor batch of images of the size [N,C_in,H_in,W_in]
  ##     - ``kernel`` 4D Tensor convolving kernel filters of the size [C_out,C_in,kH,kW]
  ##     - ``bias`` 3D Tensor bias of the size [C_out,1,1]

  # TODO bias as an Optional type

  const rank: cint = 4

  let srcTensorDesc = newCudnn4DTensorDesc    input # TODO: destructor
  let kernelDesc =    newCudnnConvKernelDesc  kernel # TODO: destructor
  let convDesc = newConv2dDesc[T](padding, strides, dilation) # TODO: destructor

  # Setting up the result
  let result_shape = convOutDims(input, kernel, padding, strides, dilation)
  result = newCudaTensor[T](result_shape)
  let dstTensorDesc = newCudnn4DTensorDesc result

  # Getting the convolution algorithm, shared memory workspace and its size.
  let algo_workspace = newConvAlgoSpace[T](srcTensorDesc, kernelDesc, convDesc, dstTensorDesc)

  # Scaling factors
  var alpha: T = 1
  var beta: T = 0

  # Pointer to workspace buffer
  # Pass nil if no workspace need to be allocated
  let pworkspace = block:
    if algo_workspace.sizeInBytes == 0:
      nil
    else:
      algo_workspace.workspace[]

  discard cudnnConvolutionForward(
    cudnnHandle0,
    addr alpha,
    srcTensorDesc,
    input.get_offset_ptr,
    kernelDesc,
    kernel.get_offset_ptr,
    convDesc,
    algo_workspace.algo,
    pworkspace,
    algo_workspace.sizeInBytes,
    addr beta,
    dstTensorDesc,
    result.get_offset_ptr
  )

  result .+= bias.unsqueeze(0)

proc conv2d_backward*[T: SomeFloat](input, kernel, bias: CudaTensor[T],
                         padding: SizeHW = [0,0],
                         strides, dilation: SizeHW = [1,1],
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
  ##     - ``strides`` SizeHW tuple with height and width of the convolution strides
  ##     - ``dilation`` SizeHW tuple with a rescaling factor of the convolution
  ##     - ``grad_output`` 4D tensor gradient of the next layer of the size [N,C_out,H_out,W_out]
  ##     - ``grad_input`` tensor where the gradient w.r.t input will be written
  ##     - ``grad_kernel`` tensor where the gradient w.r.t convolution kernel will be written
  ##     - ``grad_bias`` tensor where the gradient w.r.t bias will be written
  ##
  ## Note:
  ##   ``grad_input``, ``grad_kernel`` and ``grad_bias`` will be overwritten. They must have the same shape
  ##    as the corresponding ``input``, ``kernel`` and ``bias``
  const rank: cint = 4

  # CuDNN requires grad_output to be C contiguous. (It is undocumented as of CuDNN v7)
  # If grad_output is F contiguous it throws CUDNN_STATUS_NOT_SUPPORTED in the algo procs.
  let gOutput = grad_output.asContiguous(rowMajor, force = true)

  let # TODO: Automatic destructor
    srcTensorDesc =        newCudnn4DTensorDesc   input
    kernelDesc =           newCudnnConvKernelDesc kernel
    gradKernelDesc =       newCudnnConvKernelDesc grad_kernel
    gradInputTensorDesc =  newCudnn4DTensorDesc   grad_input
    gradOutputTensorDesc = newCudnn4DTensorDesc   gOutput
    convDesc =             newConv2dDesc[T](padding, strides, dilation) # TODO: destructor

  # Scaling factors
  var alpha: T = 1
  var beta: T = 0

  # Bias gradient
  if bias.rank > 0:
    let gradBiasTensorDesc = newCudnn4DTensorDesc grad_bias.unsqueeze(0)
    check cudnnConvolutionBackwardBias(
      cudnnHandle0,
      addr alpha,
      gradOutputTensorDesc,
      gOutput.get_offset_ptr,
      addr beta,
      gradBiasTensorDesc,
      grad_bias.get_offset_ptr
    )

    # TODO squeeze and divide by batch size?

  # Kernel: getting the backward conv algorithm, shared memory workspace and its size.
  let kernel_algo_workspace = conv_bwd_kernel_algo_workspace[T](
                                srcTensorDesc,
                                gradOutputTensorDesc,
                                gradKernelDesc, # Note the gradKernel
                                convDesc
                                )

  when defined(debug):
    echo "Launching conv2D backward for kernel"

  # Pointer to workspace buffer
  # Pass nil if no workspace need to be allocated
  let k_pworkspace = block:
    if kernel_algo_workspace.sizeInBytes == 0:
      nil
    else:
      kernel_algo_workspace.workspace[]

  # Kernel gradient
  check cudnnConvolutionBackwardFilter(
    cudnnHandle0,
    addr alpha,
    srcTensorDesc,
    input.get_offset_ptr,
    gradOutputTensorDesc,
    gOutput.get_offset_ptr,
    convDesc,
    kernel_algo_workspace.algo,
    k_pworkspace,
    kernel_algo_workspace.sizeInBytes,
    addr beta,
    gradKernelDesc,
    grad_kernel.get_offset_ptr
  )

  when defined(debug):
    echo "Finished conv2D backward for kernel"

  # Input: getting the backward conv algorithm, shared memory workspace and its size.
  let gradInput_algo_workspace = conv_bwd_data_algo_workspace[T](
                                srcTensorDesc,
                                gradOutputTensorDesc,
                                kernelDesc, # Note the kernel
                                convDesc,
                                gradInputTensorDesc
                                )

  when defined(debug):
    echo "Launching conv2D backward for input"

  # Pointer to workspace buffer
  # Pass nil if no workspace need to be allocated
  let g_pworkspace = block:
    if gradInput_algo_workspace.sizeInBytes == 0:
      nil
    else:
      gradInput_algo_workspace.workspace[]

  # Input gradient
  check cudnnConvolutionBackwardData(
    cudnnHandle0,
    addr alpha,
    kernelDesc,
    kernel.get_offset_ptr,
    gradOutputTensorDesc,
    gOutput.get_offset_ptr,
    convDesc,
    gradInput_algo_workspace.algo,
    g_pworkspace,
    gradInput_algo_workspace.sizeInBytes,
    addr beta,
    gradInputTensorDesc,
    grad_input.get_offset_ptr
  )
