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

type ReshapeGate* [TT] = ref object of Gate[TT]
  cached_input_shape: MetadataArray
  cached_output_shape: MetadataArray

method forward*[TT](self: ReshapeGate[TT], a: Variable[TT]): Variable[TT] {.inline, locks:0.}=
  new result

  result.tape = a.tape
  result.value = a.value.reshape(self.cached_output_shape)
  result.grad = zeros_like(result.value)

method backward*[TT](self: ReshapeGate[TT], gradient: TT): SmallDiffs[TT] {.noInit, inline, locks:0.}=
  result[0] = gradient.reshape(self.cached_input_shape)

proc reshapeT[TT](a: Variable[TT], shape: MetadataArray): Variable[TT] =
  # Gate
  var gate: ReshapeGate[TT]
  new gate
  gate.arity = 1
  gate.cached_input_shape = a.value.shape
  gate.cached_output_shape = shape

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


proc reshape*[TT](a: Variable[TT], shape: varargs[int]): Variable[TT] =
  ## Input:
  ##   - A variable
  ##   - A shape
  reshapeT(a, shape.toMetadataArray)


proc reshape*[TT](a: Variable[TT], shape: MetadataArray): Variable[TT] =
  ## Input:
  ##   - A variable
  ##   - A shape
  reshapeT(a, shape)

proc flatten*[TT](a: Variable[TT]): Variable[TT] =
  ## Input:
  ##   - A variable
  reshapeT(a, [a.value.shape[0], a.value.size div a.value.shape[0]].toMetadataArray)