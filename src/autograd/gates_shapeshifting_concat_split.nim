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
        ./autograd_common,
        sequtils

type StackGate{.final.}[TT] = ref object of Gate[TT]
  ## TODO support unlimited stacking
  axis: int
  nb_grads: int

proc stack_forward[TT](self: StackGate[TT], x: varargs[Variable[TT]]): Variable[TT] =
  new result

  # TODO: avoid the intermediate seq alloc to extract varargs Tensors from varargs variables
  var ts: seq[TT]
  for variable in x:
    ts.add variable.value

  result.context = x[0].context
  result.value = stack(ts, self.axis)

proc stack_backward[TT](self: StackGate[TT], payload: Payload[TT]): SmallDiffs[TT] {.noInit.}=
  let gradient = payload.variable.grad
  result = newDiffs[TT](self.nb_grads)
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

  when compileOption("boundChecks"):
    let v0 = variables[0]
    for v in variables:
      check_ctx(v0, v)
      assert v0.value.shape == v.value.shape

  # Gate
  var gate: StackGate[TT]
  new gate
  gate.nb_grads = variables.len
  gate.axis = axis

  # Resulting var
  result = gate.stack_forward variables

  # Caching for backprop
  if anyIt(variables, it.is_grad_needed):
    result.grad = zeros_like result.value
    result.requires_grad = true

    register_node(
      "Stack",
      gate,
      stack_backward[TT],
      result,
      variables
    )

# ###########################################################

type ChunkSplitGate*{.final.}[TT] = ref object of Gate[TT]
  axis: int

proc chunk_forward[TT](self: ChunkSplitGate[TT], x: Variable[TT], nb_chunks: Positive): seq[Variable[TT]] {.noInit, inline.}=
  result = x.value.chunk(nb_chunks, self.axis).mapIt( # TODO: inefficient to create an intermediate sequence
    Variable[TT](context: x.context, value: it)
  )

proc chunk_backward[TT](self: ChunkSplitGate[TT], payload: Payload[TT]): SmallDiffs[TT] {.noInit.}=
  let gradients = payload.sequence.mapIt(it.grad) # TODO: inefficient to create an intermediate sequence
  result = newDiffs[TT](1)
  result[0] = concat(gradients, self.axis)

proc chunk*[TT](v: Variable[TT], nb_chunks: Positive, axis: Natural): seq[Variable[TT]] =
  ## Splits a Variable into n chunks along the specified axis.
  ##
  ## In case a tensor cannot be split evenly,
  ## with la == length_axis, n = n_chunks
  ## it returns la mod n subtensors of size `(la div n) + 1`
  ##            the rest of size `la div n`.
  ## So split sizes at most differs by 1
  ##
  ## This is consistent with numpy array_split

  # Gate
  var gate: ChunkSplitGate[TT]
  new gate
  gate.axis = axis

  # Resulting var
  result = gate.chunk_forward(v, nb_chunks)

  # Caching for backprop
  if v.requires_grad:
    for idx, variable in result:
      variable.requires_grad = true
      variable.grad = zeros_like variable.value

    register_node(
      "Chunk",
      gate,
      chunk_backward[TT],
      result,
      v
    )
