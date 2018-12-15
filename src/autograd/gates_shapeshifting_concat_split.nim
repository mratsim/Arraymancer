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
        ../private/sequninit,
        ./autograd_common,
        sequtils

type StackGate{.final.}[TT] = ref object of Gate[TT]
  ## TODO support unlimited stacking
  axis: int
  nb_grads: int

proc stack_backward_ag[TT](self: StackGate[TT], payload: Payload[TT]): SmallDiffs[TT] =
  let gradient = payload.variable.grad
  result = newDiffs[TT](self.nb_grads)
  for i in 0 ..< gradient.shape[self.axis]:
    result[i] = gradient.atAxisIndex(self.axis, i)

proc stack_cache[TT](result: Variable[TT], variables: varargs[Variable[TT]], axis: int) =
  # Gate
  var gate: StackGate[TT]
  new gate
  gate.nb_grads = variables.len
  gate.axis = axis

  # Result setup
  result.grad = zeros_like result.value
  result.requires_grad = true

  # Add to graph
  register_node(
    "Stack",
    gate,
    stack_backward_ag[TT],
    result,
    variables
  )

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

  # Resulting var
  new result
  var ts = newSeqUninit[TT](variables.len)
  for i in 0 ..< variables.len:
    # TODO: avoid the intermediate seq alloc to extract varargs Tensors from varargs variables
    ts[i] = variables[i].value

  result.context = variables[0].context
  result.value = stack(ts, axis)

  # Caching for backprop
  if anyIt(variables, it.is_grad_needed):
    result.stack_cache(variables, axis)

# ###########################################################

type ChunkSplitGate*{.final.}[TT] = ref object of Gate[TT]
  axis: int

proc chunk_inference[TT](result: var seq[Variable[TT]], x: Variable[TT], nb_chunks: Positive, axis: int) =
  # TODO: inefficient to create an intermediate sequence
  result = x.value.chunk(nb_chunks, axis).mapIt(
    Variable[TT](context: x.context, value: it)
  )

proc chunk_backward_ag[TT](self: ChunkSplitGate[TT], payload: Payload[TT]): SmallDiffs[TT] =
  let gradients = payload.sequence.mapIt(it.grad) # TODO: inefficient to create an intermediate sequence
  result = newDiffs[TT](1)
  result[0] = concat(gradients, self.axis)

proc chunk_cache[TT](result: var seq[Variable[TT]], x: Variable[TT], nb_chunks: Positive, axis: int) =
  for idx, variable in result:
    variable.requires_grad = true
    variable.grad = zeros_like variable.value

  # Gate
  var gate: ChunkSplitGate[TT]
  new gate
  gate.axis = axis

  register_node(
    "Chunk",
    gate,
    chunk_backward_ag[TT],
    result,
    x
  )

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

  # Resulting var
  result.chunk_inference(v, nb_chunks, axis)

  # Caching for backprop
  if v.requires_grad:
    result.chunk_cache(v, nb_chunks, axis)
