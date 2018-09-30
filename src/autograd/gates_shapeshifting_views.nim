# Copyright 2017-2018 Mamy André-Ratsimbazafy & the Arraymancer contributors
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

import  ../tensor/tensor,
        ./ag_data_structure

template `[]`*[TT](v: Variable[TT], args: varargs[untyped]): Variable[TT] =
  ## Slice the tensor contained by the dynamic graph Variable
  ## Input:
  ##   - a Variable
  ## Output:
  ##   - a sliced Variable

  # TODO - investigate https://github.com/mratsim/Arraymancer/issues/241
  # As https://github.com/mratsim/Arraymancer/commit/e609e998d663710281dbe161249a0139befa818c
  # which fixed https://github.com/mratsim/Arraymancer/issues/185 had to be rollbacked

  # Ensure that v is only called once even if it's a function with side-effects
  let z = v

  # TODO: backprop support
  var result: type(z)
  new result

  result.context = z.context
  result.value = z.value[args]
  result.grad = z.grad[args]
  result.requires_grad = z.requires_grad

  result

  # TODO: tests for slicing correspondence

# #############################################

type ReshapeGate* [TT] = ref object of Gate[TT]
  cached_input_shape: MetadataArray
  cached_output_shape: MetadataArray

proc forward[TT](self: ReshapeGate[TT], a: Variable[TT]): Variable[TT] {.inline.}=
  new result

  result.context = a.context
  result.value = a.value.reshape(self.cached_output_shape)

method backward*[TT](self: ReshapeGate[TT], payload: Payload[TT]): SmallDiffs[TT] {.noInit, inline, locks:0.}=
  let gradient = payload.variable.grad
  result[0] = gradient.reshape(self.cached_input_shape)

proc reshapeImpl[TT](a: Variable[TT], shape: MetadataArray): Variable[TT] =
  # Gate
  var gate: ReshapeGate[TT]
  new gate
  gate.nb_grads = 1
  gate.cached_output_shape = shape

  # Node
  var node: Node[TT]
  new node

  node.gate = gate
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

proc reshape*[TT](a: Variable[TT], shape: varargs[int]): Variable[TT] =
  ## Input:
  ##   - A variable
  ##   - A shape
  reshapeImpl(a, shape.toMetadataArray)

proc reshape*[TT](a: Variable[TT], shape: MetadataArray): Variable[TT] =
  ## Input:
  ##   - A variable
  ##   - A shape
  reshapeImpl(a, shape)

proc flatten*[TT](a: Variable[TT]): Variable[TT] =
  ## Input:
  ##   - A variable
  reshapeImpl(a, [a.value.shape[0], a.value.size div a.value.shape[0]].toMetadataArray)
