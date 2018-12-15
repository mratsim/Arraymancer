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

# By convention a is the LHS (left-hand side)
# b is the rhs (right-hand side)

import  ../tensor/tensor,
        ./autograd_common

type AddGate* {.final.} [TT] = ref object of Gate[TT]

proc add_forward[TT](self: AddGate[TT], a, b: Variable[TT]): Variable[TT] {.inline.}=
  new result
  result.context = a.context
  result.value = a.value + b.value

proc add_backward_ag[TT](self: AddGate[TT], payload: Payload[TT]): SmallDiffs[TT] {.noInit.}=
  let gradient = payload.variable.grad
  result = newSeq[TT](2)
  result[0] = gradient
  result[1] = gradient

proc `+`*[TT](a, b: Variable[TT]): Variable[TT] =
  when compileOption("boundChecks"):
    check_ctx(a, b)

  # Gate
  var gate: AddGate[TT]
  new gate

  # Resulting var
  result = gate.add_forward(a, b)

  # Caching for backprop
  if a.is_grad_needed or b.is_grad_needed:
    result.grad = zeros_like result.value
    result.requires_grad = true

    register_node(
      "Add",
      gate,
      add_backward_ag[TT],
      result,
      a, b
    )

type SubGate* {.final.} [TT] = ref object of Gate[TT]

proc sub_forward[TT](self: SubGate[TT], a, b: Variable[TT]): Variable[TT] {.inline.}=
  new result
  result.context = a.context
  result.value = a.value - b.value

proc sub_backward[TT](self: SubGate[TT], payload: Payload[TT]): SmallDiffs[TT] {.noInit.}=
  let gradient = payload.variable.grad
  result = newDiffs[TT](2)
  result[0] = gradient
  result[1] = -gradient

proc `-`*[TT](a, b: Variable[TT]): Variable[TT] =
  when compileOption("boundChecks"):
    check_ctx(a, b)

  # Gate
  var gate: SubGate[TT]
  new gate

  # Resulting var
  result = gate.sub_forward(a, b)

  # Caching for backprop
  if a.is_grad_needed or b.is_grad_needed:
    result.grad = zeros_like result.value
    result.requires_grad = true

    register_node(
      "Sub",
      gate,
      sub_backward[TT],
      result,
      a, b
    )
