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

import  ../../tensor,
        ../../nn_primitives,
        ../../autograd

type ReluActivation*[TT] {.final.} = ref object of Gate[TT]
  cache: TT

proc relu_backward_ag[TT](self: Gate[TT], payload: Payload[TT]): SmallDiffs[TT] =
  let self = ReluActivation[TT](self)
  let gradient = payload.variable.grad
  result = newDiffs[TT](1)
  result[0] = gradient.relu_backward(self.cache)

proc relu_cache[TT](result: Variable[TT], a: Variable[TT]) =
  # Gate
  var gate: ReluActivation[TT]
  new gate
  gate.cache = result.value

  # Result setup
  result.grad = zeros_like(result.value)
  result.requires_grad = true

  # Add to graph
  register_node(
    "Relu",
    gate,
    relu_backward_ag[TT],
    result,
    a
  )

proc relu*[TT](a: Variable[TT]): Variable[TT] =
  ## Input:
  ##   - A variable

  # Resulting var
  new result
  result.context = a.context
  result.value = relu a.value

  # Caching for backprop
  if a.is_grad_needed:
    result.relu_cache(a)
