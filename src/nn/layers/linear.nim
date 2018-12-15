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
        ../../nn_primitives/nn_primitives,
        ../../autograd/autograd

type LinearGate* {.final.} [TT] = ref object of Gate[TT]
  ## TODO: use fused AddMatMul gate: C <- alpha AB + beta C
  input, weight, bias: Variable[TT]
  nb_grads: int

proc linear_forward[TT](self: LinearGate[TT], input: Variable[TT]): Variable[TT] {.inline.}=
  new result

  if self.bias.isNil:
    linear(input.value, self.weight.value, result.value)
  else:
    linear(input.value, self.weight.value, self.bias.value, result.value)

  result.context = input.context

proc linear_backward[TT](self: LinearGate[TT], payload: Payload[TT]): SmallDiffs[TT] {.noInit.}=
  # result[0] grad w.r.t. input
  # result[1] grad w.r.t. weight
  # result[2] grad w.r.t. bias

  let gradOutput = payload.variable.grad
  result = newDiffs[TT](self.nb_grads)

  if self.input.requires_grad:
    result[0] = gradOutput * self.weight.value

  if self.weight.requires_grad:
    result[1] = gradOutput.transpose * self.input.value

  if not self.bias.isNil and self.bias.requires_grad:
    result[2] = sum(gradOutput, axis = 0)

proc linear*[TT](input, weight: Variable[TT], bias: Variable[TT] = nil): Variable[TT] =
  ## Input:
  ##   - A x Variable of shape [batch_size, in_features]
  ##   - A weight Variable of shape [out_features, in_features]
  ##   - Optionally a bias Variable of shape [1, out_features]
  ##
  ## Return:
  ##   - Weight * x + bias
  ##
  ## Future TODO:
  ##   In the future the linear layer will allow different input layout
  ##   so that x can also be of shape [batch_size, in_features]
  ##
  ## Warning ⚠:
  ##  - Experimental, there is no tests yet for this layer

  when compileOption("boundChecks"):
    if input.value.rank > 2:
      raise newException(ValueError, "Tensor must be flattened for a linear layer (features, batch_size)")

    check_ctx(input, weight)
    if not bias.isNil:
      check_ctx(input, bias)

    # weight has shape: Out_features * In_features
    # bias must have shape: Out_features * 1
    if not bias.isNil and not (bias.value.shape == [1, weight.value.shape[0]].toMetadataArray):
      raise newException(ValueError, "Incompatible shape: bias must be a vector of shape [out_features, 1]")

  # Gate
  var gate: LinearGate[TT]
  new gate
  gate.nb_grads = if bias.isNil: 2 else: 3
  gate.input = input
  gate.weight = weight
  gate.bias = bias

  # Resulting var
  result = gate.linear_forward(input)

  # Caching for backprop
  if input.is_grad_needed or weight.is_grad_needed or (not bias.isNil and bias.is_grad_needed):
    result.grad = zeros_like(result.value)
    result.requires_grad = true

    if not bias.isNil:
      register_node(
        "Linear",
        gate,
        linear_backward[TT],
        result,
        input, weight, bias
      )
    else:
      register_node(
        "Linear",
        gate,
        linear_backward[TT],
        result,
        input, weight
      )
