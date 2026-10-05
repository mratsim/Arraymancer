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

type LinearGate*[TT] {.final.} = ref object of Gate[TT]
  input, weight, bias: Variable[TT]

proc linear_backward_ag[TT](self: Gate[TT], payload: Payload[TT]): SmallDiffs[TT] =
  let self = LinearGate[TT](self)
  let gradOutput = payload.variable.grad
  if self.bias.isNil:
    result = newDiffs[TT](2)
    linear_backward(self.input.value, self.weight.value, gradOutput, result[0], result[1])
  else:
    result = newDiffs[TT](3)
    linear_backward(self.input.value, self.weight.value, gradOutput, result[0], result[1], result[2])

proc linear_cache[TT](result: Variable[TT], input, weight, bias: Variable[TT]) =
  var gate: LinearGate[TT]
  new gate
  gate.input = input
  gate.weight = weight

  result.grad = zeros_like(result.value)
  result.requires_grad = true

  if not bias.isNil:
    gate.bias = bias
    register_node(
      "Linear",
      gate,
      linear_backward_ag[TT],
      result,
      input, weight, bias
    )
  else:
    register_node(
      "Linear",
      gate,
      linear_backward_ag[TT],
      result,
      input, weight
    )

proc linear*[TT](input, weight: Variable[TT], bias: Variable[TT] = nil): Variable[TT] =
  ## Input:
  ##   - An input Variable of shape [..., in_features]
  ##   - A weight Variable of shape [out_features, in_features]
  ##   - Optionally a bias Variable of shape [1, out_features]
  ##
  ## Return:
  ##   - x * Weight^T + bias

  when compileOption("boundChecks"):
    if input.value.rank < 2:
      raise newException(ValueError, "Input tensor must have rank >= 2 for linear layer")
    if input.value.shape[input.value.rank - 1] != weight.value.shape[1]:
      raise newException(ValueError, "Incompatible shape: input last dimension (" & $input.value.shape[input.value.rank - 1] & ") must match weight in_features (" & $weight.value.shape[1] & ")")

    check_ctx(input, weight)
    if not bias.isNil:
      check_ctx(input, bias)

    if not bias.isNil and not (bias.value.shape == [1, weight.value.shape[0]].toMetadata):
      raise newException(ValueError, "Incompatible shape: bias must be a vector of shape [1, out_features]")

  new result
  result.context = input.context
  if bias.isNil:
    linear(input.value, weight.value, result.value)
  else:
    linear(input.value, weight.value, bias.value, result.value)

  if input.is_grad_needed or weight.is_grad_needed or (not bias.isNil and bias.is_grad_needed):
    result.linear_cache(input, weight, bias)

type
  Linear*[T] = object
    weight*: Variable[Tensor[T]]
    bias*: Variable[Tensor[T]]

proc init*[T](
  ctx: Context[Tensor[T]],
  layerType: typedesc[Linear[T]],
  numInput, numOutput: int,
  bias: bool = true
): Linear[T] =
  ## Initializes a linear layer with `numInput` input features and `numOutput` output features.
  ## Using Kaiming He initialisation for weights.
  ## Biases are initialized to zero.
  result.weight = ctx.variable(kaiming_normal([numOutput, numInput], T), requiresGrad = true)
  if bias:
    result.bias = ctx.variable(zeros[T]([1, numOutput]), requiresGrad = true)

proc forward*[T](self: Linear[T], input: Variable[Tensor[T]]): Variable[Tensor[T]] =
  input.linear(weight = self.weight, bias = self.bias)

func outShape*[T](self: Linear[T]): seq[int] =
  @[self.weight.value.shape[0]]
func inShape*[T](self: Linear[T]): seq[int] =
  @[self.weight.value.shape[1]]
