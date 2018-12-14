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

import  ../private/ast_utils,
        ../tensor/tensor,
        ./autograd_common,
        sequtils

type MeanGate* {.final.} [TT] = ref object of Gate[TT]
  ## TODO: generalize to C <- alpha AB + C
  cached_input_shape: MetadataArray

proc forward[TT](self: MeanGate[TT], a: Variable[TT]): Variable[TT] {.inline.}=
  new result

  result.context = a.context
  result.value = [a.value.mean].toTensor

method backward*[TT](self: MeanGate[TT], payload: Payload[TT]): SmallDiffs[TT] {.noInit, inline.}=
  let gradient = payload.variable.grad
  result = newDiffs[TT](1)
  result[0] = gradient / getSubType(TT)(self.cached_input_shape.product) # Conversion to subtype T, oh Higher kinded-types ...

  let z_shape = newSeqWith(self.cached_input_shape.len, 1) # We create a shape of 1 dimension that we will expand with broadcast
  result[0] = result[0].reshape(z_shape).broadcast(self.cached_input_shape)

proc mean*[TT](a: Variable[TT]): Variable[TT] =
  # Gate
  var gate: MeanGate[TT]
  new gate

  # Node
  var node: Node[TT]
  new node

  node.gate = gate
  node.parents = newParents[TT](1)
  node.parents[0] = a.weakRef

  a.context.push(node)

  # Resulting var
  result = gate.forward(a)
  node.payload = Payload[TT](kind: pkVar, variable: result)

  # Caching for backprop
  if a.is_grad_needed:
    result.grad = zeros_like(result.value)
    result.requires_grad = true

    gate.cached_input_shape = a.value.shape

type SumGate* {.final.} [TT] = ref object of Gate[TT]
  ## TODO: generalize to C <- alpha AB + C
  cached_input_shape: MetadataArray

proc forward[TT](self: SumGate[TT], a: Variable[TT]): Variable[TT] {.inline.}=
  new result

  result.context = a.context
  result.value = [a.value.sum].toTensor

method backward*[TT](self: SumGate[TT], payload: Payload[TT]): SmallDiffs[TT] {.noInit, inline.}=
  let gradient = payload.variable.grad
  result = newDiffs[TT](1)

  let z_shape = newSeqWith(self.cached_input_shape.len, 1) # We create a shape of 1 dimension that we will expand with broadcast
  result[0] = gradient.reshape(z_shape).broadcast(self.cached_input_shape)

proc sum*[TT](a: Variable[TT]): Variable[TT] =
  # Gate
  var gate: SumGate[TT]
  new gate

  # Node
  var node: Node[TT]
  new node

  node.gate = gate
  node.parents = newParents[TT](1)
  node.parents[0] = a.weakRef

  a.context.push(node)

  # Resulting var
  result = gate.forward(a)
  node.payload = Payload[TT](kind: pkVar, variable: result)

  # Caching for backprop
  if a.is_grad_needed:
    result.grad = zeros_like(result.value)
    result.requires_grad = true

    gate.cached_input_shape = a.value.shape
