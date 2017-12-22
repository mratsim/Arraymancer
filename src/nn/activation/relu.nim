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

type ReluActivation* {.final.} [TT] = ref object of Gate[TT]
  cache: TT

method forward*[TT](self: ReluActivation[TT], a: Variable[TT]): Variable[TT] {.inline, locks:0.}=
  new result

  result.tape = a.tape
  result.value = relu a.value
  result.grad = zeros_like(result.value)

method backward*[TT](self: ReluActivation[TT], gradient: TT): SmallDiffs[TT] {.noInit, inline, locks:0.}=
  result[0] = gradient.relu_backward(self.cache)

proc relu*[TT](a: Variable[TT]): Variable[TT] =
  ## Input:
  ##   - A variable

  # Gate
  var gate: ReluActivation[TT]
  new gate
  gate.arity = 1

  # Node
  var node: Node[TT]
  new node

  node.gate = gate
  node.parents[0] = a.weakRef

  a.tape.push(node)

  # Resulting var
  result = gate.forward(a)
  node.child = result

  # Caching for backprop
  gate.cache = result.value