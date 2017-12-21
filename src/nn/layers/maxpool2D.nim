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


type MaxPool2DGate* {.final.} [TT] = ref object of Gate[TT]
  cached_input_shape: MetadataArray
  cached_max_indices: Tensor[int]
  kernel, padding, stride: Size2D

method forward*[TT](self: MaxPool2DGate[TT], a: Variable[TT]): Variable[TT] {.inline, locks:0.}=
  new result

  result.tape = a.tape
  (self.cached_max_indices, result.value) = maxpool2d(a.value,
                                                      self.kernel,
                                                      self.padding,
                                                      self.stride)
  result.grad = zeros_like(result.value)


method backward*[TT](self: MaxPool2DGate[TT], gradient: TT): SmallDiffs[TT] {.noInit, inline, locks:0.}=
  result[0] = maxpool2d_backward(
    self.cached_input_shape,
    self.cached_max_indices,
    gradient
  )

proc maxpool2d*[TT](input: Variable[TT],
                    kernel: Size2D,
                    padding: Size2D = (0,0),
                    stride: Size2D = (1,1)
                    ): Variable[TT] {.inline.} =
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

  # Gate
  var gate: MaxPool2DGate[TT]
  new gate
  gate.arity = 1
  gate.cached_input_shape = input.value.shape
  gate.kernel = kernel
  gate.padding = padding
  gate.stride = stride

  # Node
  var node: Node[TT]
  new node

  node.gate = gate
  node.parents[0] = input

  input.tape.push(node)

  # Resulting var
  result = gate.forward(input)
  result.ancestor = node
  node.child = result