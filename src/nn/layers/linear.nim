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

import ../../arraymancer_ag, ../../arraymancer, ../../autograd/utils

type LinearGate* {.final.} [TT] = ref object of Gate[TT]
  ## TODO: use fused AddMatMul gate: C <- alpha AB + beta C
  cache: TT
  weight: TT
  bias: TT
  dW: TT
  dB: TT

method forward*[TT](self: LinearGate[TT], a: Variable[TT]): Variable[TT] {.inline.}=
  new result

  result.tape = a.tape
  result.value = self.weight * a.value
  result.value .+= self.bias # Bias is broadcasted other the whole batch size
  result.grad = zeros[getSubType(TT)](result.value.shape)

method backward*[TT](self: LinearGate[TT], gradient: TT): SmallDiffs[TT] =
  self.dW = gradient * self.cache.unsafeTranspose
  self.dB = sum(gradient, axis=0) # https://mlxai.github.io/2017/01/10/a-modular-approach-to-implementing-fully-connected-neural-networks.html

  result[0] = self.weight.unsafeTranspose * gradient

proc linear*[TT](a: Variable[TT], out_shape: int): Variable[TT] =
  ## Input:
  ##   - A variable
  ##   - Number of features in the output
  ##
  # TODO: introduce initialization scheme for weight and bias
  # when compileOption("boundChecks"):
  #   check_ctx(a, b)

  # TODO: batch_size, where to put it? (out_shape, N) or (N, out_shape)

  when compileOption("boundChecks"):
    if a.value.rank > 2:
      raise newException(ValueError, "Tensor must be flattened for a linear layer (features, batch_size)")

  # Gate
  var gate: LinearGate[TT]
  new gate
  gate.arity = 1
  gate.cache = a.value.unsafeView
  gate.weight = zeros[getSubType(TT)](out_shape, a.value.shape[0])
  gate.bias = zeros[getSubType(TT)](out_shape, 1)

  # Node
  var node: Node[TT]
  new node

  node.gate = gate
  node.parents[0] = a

  a.tape.push(node)

  # Resulting var
  result = gate.forward(a)
  result.ancestor = node
  node.child = result