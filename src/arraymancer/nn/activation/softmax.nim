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

import  ../../autograd,
        ../../tensor,
        ../../nn_primitives

type SoftmaxActivation* [TT] = ref object of Gate[TT]
  cache: TT

proc softmax_backward_ag[TT](self: SoftmaxActivation[TT], payload: Payload[TT]): SmallDiffs[TT] =
  let gradient = payload.variable.grad
  result = newDiffs[TT](1)
  result[0] = gradient.softmax_backward(self.cache)

proc softmax_cache[TT](result: Variable[TT], a: Variable[TT]) =
  # Gate
  var gate: SoftmaxActivation[TT]
  new gate
  gate.cache = result.value

  # Result setup
  result.grad = zeros_like(result.value)
  result.requires_grad = true

  # Add to graph
  register_node(
    "Softmax",
    gate,
    softmax_backward_ag[TT],
    result,
    a
  )

proc softmax*[TT](a: Variable[TT]): Variable[TT] =
  ## Input:
  ##   - A variable

  # Resulting var
  new result
  result.context = a.context
  result.value = softmax a.value

  # Caching for backprop
  if a.is_grad_needed:
    result.softmax_cache(a)
