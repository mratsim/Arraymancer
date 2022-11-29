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

import  ../../tensor,
        ../../autograd,
        ../../nn_primitives


type MaxPool2DGate*[TT] {.final.} = ref object of Gate[TT]
  cached_input_shape: Metadata
  cached_max_indices: Tensor[int]
  kernel, padding, stride: Size2D

proc maxpool2D_inference[TT](
        result: Variable[TT],
        input: Variable[TT],
        kernel: Size2D,
        padding: Size2D,
        stride: Size2D
      ) =
  # TODO: change maxpool to be in-place
  var unused: Tensor[int]
  (unused, result.value) = maxpool2d(input.value,
                                kernel,
                                padding,
                                stride)

proc maxpool2D_backward_ag[TT](self: Gate[TT], payload: Payload[TT]): SmallDiffs[TT] =
  let self = MaxPool2DGate[TT](self)
  let gradient = payload.variable.grad
  result = newDiffs[TT](1)
  result[0] = maxpool2d_backward(
    self.cached_input_shape,
    self.cached_max_indices,
    gradient
  )

proc maxpool2D_forward[TT](
        result: Variable[TT],
        input: Variable[TT],
        kernel: Size2D,
        padding: Size2D,
        stride: Size2D
      ) =
  # Gate
  var gate: MaxPool2DGate[TT]
  new gate
  gate.kernel = kernel
  gate.padding = padding
  gate.stride = stride
  gate.cached_input_shape = input.value.shape

  (gate.cached_max_indices, result.value) = maxpool2d(input.value,
                                                      kernel,
                                                      padding,
                                                      stride)
  # Result setup
  result.grad = zeros_like(result.value)
  result.requires_grad = true

  # Add to graph
  register_node(
    "Maxpool2D",
    gate,
    maxpool2D_backward_ag[TT],
    result,
    input
  )

proc maxpool2d*[TT](input: Variable[TT],
                    kernel: Size2D,
                    padding: Size2D = (0,0),
                    stride: Size2D = (1,1)
                  ): Variable[TT] =
  ## Input:
  ##     - ``input`` Variable wrapping a 4D Tensor shape [N,C,H_in,W_in]
  ##     - ``kernel`` Height (kH) and width (kW) of the pooling kernel.
  ##     - ``padding`` Size2D tuple with height and width of the padding
  ##     - ``stride`` Size2D tuple with height and width of the stride
  ##
  ## Returns:
  ##     - A variable with a pooled 4D Tensor of shape [N,C,H_out,W_out], where
  ##        H_out = (H_in + (2*padding.height) - kH) / stride.height + 1
  ##        W_out = (W_in + (2*padding.width) - kW) / stride.width + 1
  ##
  ## Warning ⚠:
  ##  - Experimental, there is no tests yet for this layer

  when compileOption("boundChecks"):
    if unlikely(input.value.rank != 4):
      raise newException(ValueError, "Only 4d tensors are accepted for input and weight")

  # Resulting var
  new result
  result.context = input.context

  # Caching for backprop
  if input.is_grad_needed:
    result.maxpool2D_forward(input, kernel, padding, stride)
  else:
    result.maxpool2D_inference(input, kernel, padding, stride)

type
  MaxPool2D*[T] = object
    kernelSize: Size2D
    padding: Size2D
    stride: Size2D
    inShape: seq[int]

proc init*[T](
  ctx: Context[Tensor[T]],
  layerType: typedesc[MaxPool2D[T]],
  inShape: seq[int],
  kernelSize, padding, stride: Size2D
): MaxPool2D[T] =

  ## Creates an 2d maxpool layer.
  ## Input:
  ##     - ``inShape`` Expected shape if input in the form of ``[C, H_in, W_in]``
  ##     - ``kernelSize`` Height and width of the pooling kernel.
  ##     - ``padding`` Size2D tuple with height and width of the padding
  ##     - ``stride`` Size2D tuple with height and width of the stride
  ##
  ## Returns the created ``MaxPool2D``.


  result = MaxPool2D[T](
    kernelSize: kernelSize,
    padding: padding,
    stride: stride
  )
  assert inShape.len == 3
  result.inShape = inShape


proc forward*[T](self: MaxPool2D[T], input: Variable[Tensor[T]]): Variable[Tensor[T]] =
  input.maxpool2d(
    kernel = self.kernelSize,
    padding = self.padding,
    stride = self.stride
  )

func outShape*[T](self: MaxPool2D[T]): seq[int] =
  template C: int = self.inShape[0]
  template H: int = self.inShape[1]
  template W: int = self.inShape[2]

  template kH: int = self.kernelSize.height
  template kW: int = self.kernelSize.width
  template pH: int = self.padding.height
  template pW: int = self.padding.width
  template sH: int = self.stride.height
  template sW: int = self.stride.width

  @[
    C,
    (H + (2 * pH) - kH) div sH + 1,
    (W + (2 * pW) - kW) div sW + 1
  ]

func inShape*[T](self: MaxPool2D[T]): seq[int] =
  self.inShape
