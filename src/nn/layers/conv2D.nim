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

import  ../../tensor/tensor,
        ../../autograd/autograd,
        ../../nn_primitives/nn_primitives

type Conv2DGate* {.final.} [TT] = ref object of Gate[TT]
  cached_input: Variable[TT]
  weight, bias: Variable[TT]
  padding, stride: Size2D
  # TODO: store the algorithm (NNPACK / im2col)

proc conv2d_backward_ag[TT](self: Conv2DGate[TT], payload: Payload[TT]): SmallDiffs[TT] =
  let gradient = payload.variable.grad
  if self.bias.isNil:
    result = newDiffs[TT](2)
  else:
    result = newDiffs[TT](3)
  conv2d_backward(
    self.cached_input.value,
    self.weight.value, self.bias.value,
    self.padding, self.stride,
    gradient,
    result[0], result[1], result[2]
  )

proc conv2d_cache[TT](
      result: Variable[TT],
      input, weight, bias: Variable[TT],
      padding, stride: Size2D) =

  # Gate
  var gate: Conv2DGate[TT]
  new gate
  gate.cached_input = input
  gate.weight = weight
  gate.padding = padding
  gate.stride = stride

  # Result setup
  result.grad = zeros_like(result.value)
  result.requires_grad = true

  # Add to graph
  if not bias.isNil:
    gate.bias = bias
    register_node(
      "Conv2D",
      gate,
      conv2d_backward_ag[TT],
      result,
      input, weight, bias
    )
  else:
    register_node(
      "Conv2D",
      gate,
      conv2d_backward_ag[TT],
      result,
      input, weight
    )

proc conv2d*[TT]( input, weight: Variable[TT],
                  bias: Variable[TT] = nil,
                  padding: Size2D = (0,0),
                  stride: Size2D = (1,1)): Variable[TT] =
  ## Input:
  ##     - ``input`` Variable wrapping a 4D Tensor batch of images of the size [N,C_in,H_in,W_in]
  ##     - ``weight`` Variable wrapping a 4D Tensor convolving kernel weights of the size [C_out,C_in,kH,kW]
  ##     - ``bias`` Nil-able Variable wrapping a 3D Tensor bias of the size [C_out,1,1]
  ##     - ``padding`` Size2D tuple with height and width of the padding
  ##     - ``stride`` Size2D tuple with height and width of the stride
  ##
  ## Returns:
  ##     - A variable with a convolved 4D Tensor of size [N,C_out,H_out,W_out], where
  ##        H_out = (H_in + (2*padding.height) - kH) / stride.height + 1
  ##        W_out = (W_in + (2*padding.width) - kW) / stride.width + 1
  ##
  ## Future TODO:
  ##   In the future the conv2D layer will allow different input layout
  ##
  ## Warning ⚠:
  ##  - Experimental, there is no tests yet for this layer

  when compileOption("boundChecks"):
    if unlikely(input.value.rank != 4 or weight.value.rank != 4):
      raise newException(ValueError, "Only 4d tensors are accepted for input and weight")

    check_ctx(input, weight)
    if not bias.isNil:
      check_ctx(input, bias)

    # weight has shape: Out_features * In_features
    # bias must have shape: Out_features * 1
    if unlikely(not bias.isNil and bias.value.rank != 3) :
      raise newException(ValueError, "Incompatible shape: bias must be of rank 3")

  # Resulting var
  new result
  result.context = input.context
  result.value = conv2D(input.value,
                        weight.value,
                        bias.value, # Todo, case when there is no bias
                        padding,
                        stride
                      )

  # Caching for backprop:
  if input.is_grad_needed or weight.is_grad_needed or (not bias.isNil and bias.is_grad_needed):
    conv2D_cache(
        result,
        input, weight, bias,
        padding, stride
      )
