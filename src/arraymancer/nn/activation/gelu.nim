# Copyright 2026 the Arraymancer contributors
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

import  ../../tensor,
        ../../nn_primitives,
        ../../autograd

type GeluActivation*[TT] {.final.} = ref object of Gate[TT]
  cache: TT
  approximate: bool

proc gelu_backward_ag[TT](self: Gate[TT], payload: Payload[TT]): SmallDiffs[TT] =
  let self = GeluActivation[TT](self)
  let gradient = payload.variable.grad
  result = newDiffs[TT](1)
  result[0] =
    if self.approximate: gradient.quick_gelu_backward(self.cache)
    else: gradient.gelu_backward(self.cache)

proc gelu_cache[TT](result: Variable[TT], a: Variable[TT], approximate: bool) =
  # Gate
  var gate: GeluActivation[TT]
  new gate
  gate.cache = a.value
  gate.approximate = approximate

  # Result setup
  result.grad = zeros_like(result.value)
  result.requires_grad = true

  # Add to graph
  register_node(
    "Gelu",
    gate,
    gelu_backward_ag[TT],
    result,
    a
  )

proc gelu*[TT](a: Variable[TT], approximate = false): Variable[TT] =
  ## Input:
  ##   - A variable
  ## `approximate = true` uses the QuickGELU sigmoid approximation.

  # Resulting var
  new result
  result.context = a.context
  result.value = if approximate: quick_gelu a.value else: gelu a.value

  # Caching for backprop
  if a.is_grad_needed:
    result.gelu_cache(a, approximate)

proc quick_gelu*[TT](a: Variable[TT]): Variable[TT] =
  gelu(a, approximate = true)
