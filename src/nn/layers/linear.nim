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
        ../../autograd/autograd,
        ./layer



type LinearGate* {.final.} [TT] = ref object of Gate[TT]
  ## TODO: use fused AddMatMul gate: C <- alpha AB + beta C
  input, weight, bias: Variable[TT]

method forward*[TT](self: LinearGate[TT], input: Variable[TT]): Variable[TT] {.inline, locks:0.}=
  new result

  if self.bias.isNil:
    linear(input.value, self.weight.value, result.value)
  else:
    linear(input.value, self.weight.value, self.bias.value, result.value)

  result.tape = input.tape
  result.grad = zeros_like(result.value)

method backward*[TT](self: LinearGate[TT], gradOutput: TT): SmallDiffs[TT] {.noInit, inline, locks:0.}=
  # result[0] grad w.r.t. input
  # result[1] grad w.r.t. weight
  # result[2] grad w.r.t. bias

  if self.bias.isNil:
    linear_backward(
      self.input.value,
      self.weight.value,
      gradOutput,
      result[0],
      result[1]
    )
  else:
    linear_backward(
      self.input.value,
      self.weight.value,
      self.bias.value,
      gradOutput,
      result[0],
      result[1],
      result[2]
    )

proc linear*[TT](input, weight: Variable[TT], bias: Variable[TT] = nil): Variable[TT] =
  ## Input:
  ##   - A x Variable of shape [in_features, batch_size]
  ##   - A weight Variable of shape [out_features, in_features]
  ##   - Optionally a bias Variable of shape [out_features, 1]
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
    if not bias.isNil and not (bias.value.shape == [weight.value.shape[0], 1].toMetadataArray):
      raise newException(ValueError, "Incompatible shape: bias must be a vector of shape [out_features, 1]")

  # Gate
  var gate: LinearGate[TT]
  new gate
  gate.arity = if bias.isNil: 2 else: 3
  gate.input = input
  gate.weight = weight
  gate.bias = bias

  # Node
  var node: Node[TT]
  new node

  node.gate = gate
  node.parents[0] = input
  node.parents[1] = weight
  if not bias.isNil:
    node.parents[2] = bias

  input.tape.push(node)

  # Resulting var
  result = gate.forward(input)
  node.child = result