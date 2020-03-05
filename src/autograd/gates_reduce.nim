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

type MeanGate*[TT] {.final.} = ref object of Gate[TT]
  ## TODO: generalize to C <- alpha AB + C
  cached_input_shape: Metadata
  axis: int

proc shape_product(m: MeanGate): int {.inline.} =
  result = 1
  for i, v in m.cached_input_shape:
    if i != m.axis:
      result *= v

proc mean_backward_ag[TT](self: MeanGate[TT], payload: Payload[TT]): SmallDiffs[TT] =
  let gradient = payload.variable.grad
  result = newDiffs[TT](1)
  result[0] = gradient / getSubType(TT)(self.cached_input_shape.product) # Conversion to subtype T, oh Higher kinded-types ...

  # We create a shape of 1 dimension that we will expand with broadcast
  let z_shape = newSeqWith(self.cached_input_shape.len, 1)
  result[0] = result[0].reshape(z_shape).broadcast(self.cached_input_shape)

proc mean_with_axis_backward_ag[TT](self: MeanGate[TT], payload: Payload[TT]): SmallDiffs[TT] =
  let gradient = payload.variable.grad
  result = newDiffs[TT](1)
  result[0] = (gradient / getSubType(TT)(self.shape_product)).broadcast(self.cached_input_shape)

proc mean_cache[TT](result: Variable[TT], a: Variable[TT]) =
  # Gate
  var gate: MeanGate[TT]
  new gate
  gate.cached_input_shape = a.value.shape

  # Result setup
  result.grad = zeros_like(result.value)
  result.requires_grad = true

  # Caching for backprop
  register_node(
    "Mean",
    gate,
    mean_backward_ag[TT],
    result,
    a
  )

proc mean_cache[TT](result: Variable[TT], a: Variable[TT], axis: Natural) =
  # Gate
  var gate: MeanGate[TT]
  new gate
  gate.cached_input_shape = a.value.shape
  gate.axis = axis

  # Result setup
  result.grad = zeros_like(result.value)
  result.requires_grad = true

  # Caching for backprop
  register_node(
    "Mean",
    gate,
    mean_with_axis_backward_ag[TT],
    result,
    a
  )

proc mean*[TT](a: Variable[TT]): Variable[TT] =
  # Resulting var
  new result
  result.context = a.context
  result.value = [a.value.mean].toTensor

  # Caching for backprop
  if a.is_grad_needed:
    result.mean_cache(a)

proc mean*[TT](a: Variable[TT], axis: Natural): Variable[TT] =
  # Resulting var
  new result
  result.context = a.context
  result.value = a.value.mean(axis)

  # Caching for backprop
  if a.is_grad_needed:
    result.mean_cache(a, axis)

type SumGate*[TT]{.final.}= ref object of Gate[TT]
  ## TODO: generalize to C <- alpha AB + C
  cached_input_shape: Metadata

proc sum_backward_ag[TT](self: SumGate[TT], payload: Payload[TT]): SmallDiffs[TT] =
  let gradient = payload.variable.grad
  result = newDiffs[TT](1)

  # We create a shape of 1 dimension that we will expand with broadcast
  let z_shape = newSeqWith(self.cached_input_shape.len, 1)
  result[0] = gradient.reshape(z_shape).broadcast(self.cached_input_shape)

proc sum_cache[TT](result: Variable[TT], a: Variable[TT]) =
  # Gate
  var gate: SumGate[TT]
  new gate
  gate.cached_input_shape = a.value.shape

  # Result setup
  result.grad = zeros_like(result.value)
  result.requires_grad = true

  # Add to graph
  register_node(
    "Sum",
    gate,
    sum_backward_ag[TT],
    result,
    a
  )

proc sum*[TT](a: Variable[TT]): Variable[TT] =
  # Gate
  var gate: SumGate[TT]
  new gate

  # Resulting var
  new result
  result.context = a.context
  result.value = [a.value.sum].toTensor

  # Caching for backprop
  if a.is_grad_needed:
    result.sum_cache(a)
