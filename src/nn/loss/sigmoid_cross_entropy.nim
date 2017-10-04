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
import ../../arraymancer_nn_primitives

import ./loss

type SigmoidCrossEntropyLoss* {.final.} [TT] = ref object of Loss[TT]
  cache: Variable[TT]
  # arity, from Gate
  # target, from Loss

method forward*[TT](self: SigmoidCrossEntropyLoss[TT], a: Variable[TT], target: TT): Variable[TT] {.inline, locks:0.}=
  new result

  result.tape = a.tape
  # We expect a in shape @[features, batch_size]
  result.value = [sigmoid_cross_entropy(a.value, target)].toTensor

  result.grad = zeros[getSubType(TT)](1)


method backward*[TT](self: SigmoidCrossEntropyLoss[TT], gradient: TT): SmallDiffs[TT] {.inline, locks:0.}=
  result[0] = sigmoid_cross_entropy_backward(gradient, self.cache.value, self.target)

proc sigmoid_cross_entropy*[TT](a: Variable[TT], target: TT): Variable[TT] =
  # Gate
  var gate: SigmoidCrossEntropyLoss[TT]
  new gate
  gate.arity = 1
  gate.cache = a
  gate.target = target

  # Node
  var node: Node[TT]
  new node

  node.gate = gate
  node.parents[0] = a

  a.tape.push(node)

  # Resulting var
  result = gate.forward(a, target)
  result.ancestor = node
  node.child = result