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

proc conv2d*[T: SomeReal](input, convKernel, bias: CudaTensor[T],
                padding: SizeHW = [0.cint,0],
                convStrides: SizeHW = [1.cint,1]): CudaTensor[T] {.noInit.}=
  ## Input:
  ##     - ``input`` 4D Tensor batch of images of the size [N,C_in,H_in,W_in]
  ##     - ``convKernel`` 4D Tensor convolving kernel filters of the size [C_out,C_in,kH,kW]
  ##     - ``bias`` 3D Tensor bias of the size [C_out,1,1]

  # TODO bias as an Optional type

  const convDims: cint = 2
  const rank: cint = 4
  let srcTensorDesc = newCudnn4DTensorDesc input # TODO: finalizer to destroy descriptor with GC
  let dilation: SizeHW = [1.cint, 1]
  let convKernelDesc = newCudnnConvKernelDesc(convKernel) # TODO: finalizer to destroy descriptor with GC

  var convDesc: cudnnConvolutionDescriptor_t # TODO: finalizer to destroy descriptor with GC
  check cudnnCreateConvolutionDescriptor(addr convDesc)

  check cudnnSetConvolutionNdDescriptor(
    convDesc,
    convDims,
    padding.getPtr,
    convStrides.getPtr,
    dilation.getPtr,
    CUDNN_CROSS_CORRELATION,
    T.asCudnnType
  )

  # Setting up the result
  let result_shape = convOutDims(input, convKernel, padding, convStrides, dilation)
  result = newCudaTensor[T](result_shape, rowMajor)
  let dstTensorDesc = newCudnn4DTensorDesc result

  # Getting the convolution algorithm, shared memory workspace and its size.
  let algo_workspace = conv_algo_workspace[T](srcTensorDesc, convKernelDesc, convDesc, dstTensorDesc)

  # Scaling factors
  var alpha: T = 1
  var beta: T = 0

  check cudnnConvolutionForward(
    defaultHandle_cudnn,
    addr alpha,
    srcTensorDesc,
    input.data.data[],
    convKernelDesc,
    convKernel.data.data[],
    convDesc,
    algo_workspace.algo,
    algo_workspace.workspace[],
    algo_workspace.sizeInBytes,
    addr beta,
    dstTensorDesc,
    result.data.data[]
  )

  result .+= bias.unsafeUnsqueeze(0)