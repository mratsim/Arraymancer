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

import  ../../autograd/autograd,
        ../../tensor/tensor,
        ../../nn_primitives/nn_primitives,
        ../../private/ast_utils

type SigmoidActivation* {.final.} [TT] = ref object of Gate[TT]
  cache: TT

method forward*[TT](self: SigmoidActivation[TT], a: Variable[TT]): Variable[TT] {.inline, locks:0.}=
  new result

  result.tape = a.tape
  result.value = sigmoid a.value
  result.grad = zeros_like(result.value)

method backward*[TT](self: SigmoidActivation[TT], gradient: TT): SmallDiffs[TT] {.noInit, inline, locks:0.}=
  result[0] = gradient.sigmoid_backward(self.cache)

proc sigmoid*[TT](a: Variable[TT]): Variable[TT] =
  ## Input:
  ##   - A variable

  # Gate
  var gate: SigmoidActivation[TT]
  new gate
  gate.arity = 1

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

  # Caching for backprop
  gate.cache = result.value