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
        ../../nn_primitives,
        ../../autograd,
        ../init

type GCNGate*[TT] {.final.} = ref object of Gate[TT]
  adjacency, input, weight, bias: Variable[TT]

proc gcn_backward_ag[TT](self: Gate[TT], payload: Payload[TT]): SmallDiffs[TT] =
  let self = GCNGate[TT](self)
  let gradOutput = payload.variable.grad
  if self.bias.isNil:
    result = newDiffs[TT](2)
  else:
    result = newDiffs[TT](3)

  if self.input.requires_grad:
    result[0] = gradOutput * self.weight.value

  if self.weight.requires_grad:
    result[1] = gradOutput.transpose * (self.adjacency.value * self.input.value)

  if not self.bias.isNil and self.bias.requires_grad:
    result[2] = sum(gradOutput, axis = 0)

proc gcn_cache[TT](result: Variable[TT], adjacency, input, weight, bias: Variable[TT]) =
  # Gate
  var gate: GCNGate[TT]
  new gate
  gate.adjacency = adjacency
  gate.input = input
  gate.weight = weight

  # Result setup
  result.grad = zeros_like(result.value)
  result.requires_grad = true

  # Add to graph
  if not bias.isNil:
    gate.bias = bias
    register_node(
      "Linear",
      gate,
      gcn_backward_ag[TT],
      result,
      input, weight, bias
    )
  else:
    register_node(
      "Linear",
      gate,
      gcn_backward_ag[TT],
      result,
      input, weight
    )

proc gcn*[TT](input, adjacency, weight: Variable[TT], bias: Variable[TT] = nil): Variable[TT] =
  ## Input:
  ##   - A x Variable of shape [nodes, in_features]
  ##   - An adjacency matrix of shape [nodes, nodes]
  ##   - A weight Variable of shape [out_features, in_features]
  ##   - Optionally a bias Variable of shape [1, out_features]
  ##
  ## Return:
  ##   - (AX)W+b
  ##
  ##
  when compileOption("boundChecks"):
    if input.value.rank > 2:
      raise newException(ValueError, "Tensor must be flattened for a linear layer (features, batch_size)")

    check_ctx(input, weight)
    if not bias.isNil:
      check_ctx(input, bias)

    # weight has shape: Out_features * In_features
    # bias must have shape: Out_features * 1
    if not bias.isNil and not (bias.value.shape == [1, weight.value.shape[0]].toMetadata):
      raise newException(ValueError, "Incompatible shape: bias must be a vector of shape [out_features, 1]")

  # Resulting var
  new result
  result.context = input.context
  if bias.isNil:
    linear(adjacency.value * input.value, weight.value, result.value)
  else:
    linear(adjacency.value * input.value, weight.value, bias.value, result.value)

  # Caching for backprop
  if input.is_grad_needed or weight.is_grad_needed or (not bias.isNil and bias.is_grad_needed):
    result.gcn_cache(adjacency, input, weight, bias)

type
  GCNLayer*[T] = object
    weight*: Variable[Tensor[T]]
    bias*: Variable[Tensor[T]]

proc init*[T](
  ctx: Context[Tensor[T]],
  layerType: typedesc[GCNLayer[T]],
  numInput, numOutput: int
): GCNLayer[T] =
  ## Initializes a graph convolutional layer with `num_input` input features and `num_output` output features.
  ## Using Kaiming He initialisation for weights to provide decent performance in most cases.
  ## Biases are set to zero.

  result.weight = ctx.variable(kaimingNormal([numOutput, numInput], T), requiresGrad = true) # TODO allow freezing
  result.bias = ctx.variable(zeros[T]([1, numOutput]), requiresGrad = true) # TODO allow freezing

proc forward*[T](self: GCNLayer[T], input, adjacency: Variable[Tensor[T]]): Variable[Tensor[T]] =
  input.gcn(adjacency = adjacency, weight = self.weight, bias = self.bias)

proc outShape*[T](self: GCNLayer[T]): seq[int] =
  @[self.weight.value.shape[0]]
proc inShape*[T](self: GCNLayer[T]): seq[int] =
  @[self.weight.value.shape[1]]