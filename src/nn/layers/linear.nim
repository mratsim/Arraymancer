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

import  ../../private/ast_utils,
        ../../tensor/tensor,
        ../../autograd/autograd,
        ./layer



type LinearGate* {.final.} [TT] = ref object of Gate[TT]
  ## TODO: use fused AddMatMul gate: C <- alpha AB + beta C
  x, W, b: Variable[TT]

method forward*[TT](self: LinearGate[TT], a: Variable[TT]): Variable[TT] {.inline, locks:0.}=
  new result

  result.tape = a.tape
  result.value = self.W.value * a.value
  if not self.b.isNil:
    result.value .+= self.b.value # Bias is broadcasted other the whole batch size
  result.grad = zeros_like(result.value)

method backward*[TT](self: LinearGate[TT], gradient: TT): SmallDiffs[TT] {.noInit, inline, locks:0.}=
  result[0] = self.W.value.transpose * gradient # grad w.r.t. x
  result[1] = gradient * self.x.value.transpose # grad w.r.t. weight

  if not self.b.isNil:
    result[2] = sum(gradient, axis=0) # grad w.r.t. bias
    # https://mlxai.github.io/2017/01/10/a-modular-approach-to-implementing-fully-connected-neural-networks.html

proc linear*[TT](x, weight: Variable[TT], bias: Variable[TT] = nil): Variable[TT] =
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
    if x.value.rank > 2:
      raise newException(ValueError, "Tensor must be flattened for a linear layer (features, batch_size)")

    check_ctx(x, weight)
    if not bias.isNil:
      check_ctx(x, bias)

    # weight has shape: Out_features * In_features
    # bias must have shape: Out_features * 1
    if not bias.isNil and not (bias.value.shape == [weight.value.shape[0], 1].toMetadataArray):
      raise newException(ValueError, "Incompatible shape: bias must be a vector of shape [out_features, 1]")

  # Gate
  var gate: LinearGate[TT]
  new gate
  gate.arity = if bias.isNil: 2 else: 3
  gate.x = x
  gate.W = weight
  gate.b = bias

  # Node
  var node: Node[TT]
  new node

  node.gate = gate
  node.parents[0] = x
  node.parents[1] = weight
  if not bias.isNil:
    node.parents[2] = bias

  x.tape.push(node)

  # Resulting var
  result = gate.forward(x)
  result.ancestor = node
  node.child = result