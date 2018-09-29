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

import  ../private/ast_utils,
        ../tensor/tensor,
        ./ag_data_structure,
        sequtils

type StackGate{.final.}[TT] = ref object of Gate[TT]
  ## TODO support unlimited stacking
  axis: int

proc forward[TT](self: StackGate[TT], x: varargs[Variable[TT]]): Variable[TT] =
  new result

  # TODO: avoid the intermediate seq alloc to extract varargs Tensors from varargs variables
  var ts: seq[TT]
  for variable in x:
    ts.add variable.value

  result.context = x[0].context
  result.value = stack(ts, self.axis)

method backward*[TT](self: StackGate[TT], payload: Payload[TT]): SmallDiffs[TT] {.noInit, inline, locks:0.}=
  let gradient = payload.variable.grad
  for i in 0 ..< gradient.shape[self.axis]:
    result[i] = gradient.atAxisIndex(self.axis, i)

proc stack*[TT](variables: varargs[Variable[TT]], axis = 0): Variable[TT] =
  ## Join a sequence of Variables along a new axis into a new Variable.
  ## All variables must be of the same shape
  ##
  ## Input:
  ##   - a variable
  ##   - an axis (dimension)
  ## Returns:
  ##   - a new stacked variable along the new axis

  let v0 = variables[0]
  when compileOption("boundChecks"):
    for v in variables:
      check_ctx(v0, v)
      assert v0.value.shape == v.value.shape

  # Gate
  var gate: StackGate[TT]
  new gate
  gate.nb_grads = variables.len # TODO Max stack supported is 7 at the moment. Need "infinite" for timeseries and NLP
  gate.axis = axis

  # Node
  var node: Node[TT]
  new node

  node.gate = gate
  for idx, v in variables:
    node.parents[idx] = v.weakRef

  v0.context.push node

  # Resulting var
  result = gate.forward variables
  node.payload = Payload[TT](kind: pkVar, variable: result)

  # Caching for backprop
  if anyIt(variables, it.is_grad_needed):
    result.grad = zeros[getSubType(TT)](result.value.shape)
    result.requires_grad = true

type ChunkSplitGate*[TT] = ref object of Gate[TT]
  axis: int

proc forward_chunk[TT](self: ChunkSplitGate[TT], x: Variable[TT], nb_chunks: Positive): seq[Variable[TT]] {.inline.}=
  result = x.data.chunk(nb_chunks, self.axis).mapIt(ctx.variable)

method backward[TT](self: ChunkSplitGate[TT], payload: Payload[TT]): SmallDiffs[TT] =
  let gradients = payload.sequence
  var tensors = newSeq[TT](gradients.len) # TODO, inefficient to build a temp seq
  for i, val in gradients:
    tensors[i] = val.grad
  result[0] = concat(tensors, self.axis)

proc chunk*[TT](v: Variable[TT], nb_chunks: Positive, axis: Natural): seq[Variable[TT]] =
  ## Splits a Variable into n chunks along the specified axis.
  ##
  ## In case a tensor cannot be split evenly,
  ## with la == length_axis, n = n_chunks
  ## it returns la mod n subtensors of size `(la div n) + 1`
  ##            the rest of size `la div n`.
  ##
  ## This is consistent with numpy array_split

  # Gate
  var gate: ChunkSplitGate[TT]
  new gate
  gate.nb_grads = 1
  gate.axis = axis

  # Node
  var node: Node[TT]
  new node

  node.gate = gate
  node.parents[0] = v.weakRef
  v.context.push node

  # Resulting var
  result = gate.forward_chunk v
  node.payload = Payload[TT](kind: pkVar, sequence: result)

  # Caching for backprop
  if v.requires_grad:
    gate.cache = result
    gate.index_splits = newSeqUninitialized[int](nb_chunks)
    for idx, variable in result:
      variable.requires_grad = true
      variable.grad = zeros_like variable.value
