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

import  ../tensor/tensor,
        ./private/p_nnp_types,
        ./fallback/conv

when defined(nnpack):
  import backend/nnpack_interface

type
  ## Algorithms to be used in Conv2D
  Conv2DAlgorithm* = enum
    Im2ColGEMM,
    NNPackAuto

proc conv2d*[T](input, weight, bias: Tensor[T],
                padding: Size2D = (0,0),
                stride: Size2D = (1,1),
                algorithm = Conv2DAlgorithm.Im2ColGEMM): Tensor[T] {.inline.} =
  ## Computes a 2D convolution over input images. Intended to be used
  ## in 2d convolution forward pass. This applies a 2D cross-correlation,
  ## not to be confused with the mathematical convolution.
  ##
  ## Input:
  ##     - ``input`` 4D Tensor batch of images of the size [N,C_in,H_in,W_in]
  ##     - ``weight`` 4D Tensor convolving kernel weights of the size [C_out,C_in,kH,kW]
  ##     - ``bias`` 3D Tensor bias of the size [C_out,1,1] or an empty tensor for no bias
  ##     - ``padding`` Size2D tuple with height and width of the padding
  ##     - ``stride`` Size2D tuple with height and width of the stride
  ##     - ``algorithm`` algorithm to be used in the convolution
  ## Returns:
  ##     - A 4D Tensor of sized [N,C_out,H_out,W_out], where
  ##        H_out = (H_in + (2*padding.height) - kH) / stride.height + 1
  ##        W_out = (W_in + (2*padding.width) - kW) / stride.width + 1
  ## Valid algorithms:
  ##    - ``Im2ColGEMM`` im2col + GEMM algorithm, this is the default
  ##    - ``NNPackAuto`` Use NNPack and let it auto detect the best algorithm
  ##
  ## Future:
  ##    bias will leverage the upcoming Optional type to be really optional.
  assert input.rank == 4 and weight.rank == 4
  assert bias.rank == 3 or bias.rank == 0 # TODO make bias truly optional and not just a tensor of rank 0

  case algorithm:
    of NNPackAuto:
      when defined(nnpack) and T is float32:
        result = nnpack_conv2d(input, weight, bias, padding, stride)
      else:
        raise newException(LibraryError, "NNPack not enabled, enable with -d:nnpack")
    of Im2ColGEMM:
      result = im2colgemm_conv2d(input, weight, bias, padding, stride)

proc conv2d_backward*[T](input, weight, bias: Tensor[T],
                         padding: Size2D,
                         stride: Size2D,
                         grad_output: Tensor[T],
                         grad_input, grad_weight, grad_bias: var Tensor[T],
                         algorithm = Conv2DAlgorithm.Im2ColGEMM) =
  ## Computes gradients of a 2D convolution. Intended to be used after
  ## ``conv2d`` to calculate gradients in backward pass.
  ##
  ## Input:
  ##     - ``input`` 4D Tensor batch of images of the size [N,C_in,H_in,W_in]
  ##     - ``weight`` 4D Tensor convolving kernel weights of the size [C_out,C_in,kH,kW]
  ##     - ``bias`` 3D Tensor bias of the size [C_out,1,1] or an empty tensor for no bias
  ##     - ``padding`` Size2D tuple with height and width of the padding
  ##     - ``stride`` Size2D tuple with height and width of the stride
  ##     - ``grad_output`` 4D tensor gradient of the next layer of the size [N,C_out,H_out,W_out]
  ##     - ``grad_input`` tensor where the gradient w.r.t input will be written
  ##     - ``grad_weight`` tensor where the gradient w.r.t weight will be written
  ##     - ``grad_bias`` tensor where the gradient w.r.t bias will be written
  ##     - ``algorithm`` algorithm to be used in the convolution
  ## Valid algorithms:
  ##    - ``Im2ColGEMM`` im2col + GEMM algorithm, this is the default
  ##    - ``NNPackAuto`` Use NNPack and let it auto detect the best algorithm
  assert input.rank == 4 and weight.rank == 4
  assert bias.rank == 3 or bias.rank == 0

  # Bias gradient
  if bias.rank > 0: # TODO make bias truly optional and not just a tensor of rank 0
    # TODO: sum over many axes
    grad_bias = grad_output.sum(3).sum(2).sum(0).reshape(bias.shape)

  case algorithm:
    of NNPackAuto:
      when defined(nnpack) and T is float32:
        nnpack_conv2d_gradient(input, weight,
                               padding, stride,
                               grad_output, grad_input, grad_weight)
      else:
        raise newException(LibraryError, "NNPack not enabled, enable with -d:nnpack")
    of Im2ColGEMM:
      im2colgemm_conv2d_gradient(input, weight,
                                 padding, stride,
                                 grad_output, grad_input, grad_weight)
