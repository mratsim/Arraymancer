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

import  ../tensor,
        ./autograd_common,
        ./private/p_broadcast

type AddGate*[TT] {.final.} = ref object of Gate[TT]

proc add_backward_ag[TT](self: Gate[TT], payload: Payload[TT]): SmallDiffs[TT] =
  let gradient = payload.variable.grad
  result = newSeq[TT](2)
  result[0] = gradient
  result[1] = gradient

proc add_cache[TT](result: Variable[TT], a, b: Variable[TT]) =
  # Gate
  var gate: AddGate[TT]
  new gate

  # Result setup
  result.grad = zeros_like result.value
  result.requires_grad = true

  # Add to graph
  register_node(
    "Add",
    gate,
    add_backward_ag[TT],
    result,
    a, b
  )

proc `+`*[TT](a, b: Variable[TT]): Variable[TT] =
  when compileOption("boundChecks"):
    check_ctx(a, b)

  # Resulting var
  new result
  result.context = a.context
  result.value = a.value + b.value

  # Caching for backprop
  if a.is_grad_needed or b.is_grad_needed:
    result.add_cache(a, b)

type SubGate*[TT] {.final.} = ref object of Gate[TT]

proc sub_backward_ag[TT](self: Gate[TT], payload: Payload[TT]): SmallDiffs[TT] =
  # NOTE: we do ``not`` convert `self` to `SubGate` here, as that leads to an `ObjectConversionError`!
  let gradient = payload.variable.grad
  result = newSeq[TT](2)
  result[0] = gradient
  result[1] = -gradient

proc sub_cache[TT](result: Variable[TT], a, b: Variable[TT]) =
  # Gate
  var gate: AddGate[TT]
  new gate

  # Result setup
  result.grad = zeros_like result.value
  result.requires_grad = true

  # Caching for backprop
  register_node(
    "Sub",
    gate,
    sub_backward_ag[TT],
    result,
    a, b
  )

proc `-`*[TT](a, b: Variable[TT]): Variable[TT] =
  when compileOption("boundChecks"):
    check_ctx(a, b)

  # Resulting var
  new result
  result.context = a.context
  result.value = a.value - b.value

  # Caching for backprop
  if a.is_grad_needed or b.is_grad_needed:
    result.sub_cache(a, b)

type DivGate*[TT] {.final.} = ref object of Gate[TT]
  a: Variable[TT]
  b: Variable[TT]

proc div_backward_ag[TT](self: Gate[TT], payload: Payload[TT]): SmallDiffs[TT] =
  let self = DivGate[TT](self)
  let gradient = payload.variable.grad
  result = newDiffs[TT](2)
  result[0] = gradient /. self.b.value
  result[1] = - gradient *. self.a.value /. self.b.value ^. 2

proc div_cache[TT](result: Variable[TT], a, b: Variable[TT]) =
  # Gate
  var gate: DivGate[TT]
  new gate
  gate.a = a
  gate.b = b

  # Result setup
  result.grad = zeros_like result.value
  result.requires_grad = true

  # Add to graph
  register_node(
    "Div",
    gate,
    div_backward_ag[TT],
    result,
    a, b
  )

proc `/.`*[TT](a, b: Variable[TT]): Variable[TT] =
  when compileOption("boundChecks"):
    check_ctx(a, b)

  new result
  result.context = a.context
  result.value = a.value /. b.value

  if a.is_grad_needed or b.is_grad_needed:
    result.div_cache(a, b)

# ############################################################
#
#             Broadcasted Addition & Subtraction
#
# ############################################################

type AddBroadcastGate*[TT] {.final.} = ref object of Gate[TT]
  a, b: Variable[TT]

proc add_broadcast_backward_ag[TT](self: Gate[TT], payload: Payload[TT]): SmallDiffs[TT] =
  let self = AddBroadcastGate[TT](self)
  let gradient = payload.variable.grad
  result = newSeq[TT](2)
  if self.a.requires_grad:
    result[0] = reduce_broadcast_dims(gradient, self.a.value.shape)
  if self.b.requires_grad:
    result[1] = reduce_broadcast_dims(gradient, self.b.value.shape)

proc add_broadcast_cache[TT](result: Variable[TT], a, b: Variable[TT]) =
  var gate: AddBroadcastGate[TT]
  new gate
  gate.a = a
  gate.b = b
  result.grad = zeros_like result.value
  result.requires_grad = true
  register_node("AddBroadcast", gate, add_broadcast_backward_ag[TT], result, a, b)

proc `+.`*[TT](a, b: Variable[TT]): Variable[TT] =
  when compileOption("boundChecks"):
    check_ctx(a, b)
  new result
  result.context = a.context
  result.value = a.value +. b.value
  if a.is_grad_needed or b.is_grad_needed:
    result.add_broadcast_cache(a, b)

type SubBroadcastGate*[TT] {.final.} = ref object of Gate[TT]
  a, b: Variable[TT]

proc sub_broadcast_backward_ag[TT](self: Gate[TT], payload: Payload[TT]): SmallDiffs[TT] =
  let self = SubBroadcastGate[TT](self)
  let gradient = payload.variable.grad
  result = newSeq[TT](2)
  if self.a.requires_grad:
    result[0] = reduce_broadcast_dims(gradient, self.a.value.shape)
  if self.b.requires_grad:
    result[1] = reduce_broadcast_dims(-gradient, self.b.value.shape)

proc sub_broadcast_cache[TT](result: Variable[TT], a, b: Variable[TT]) =
  var gate: SubBroadcastGate[TT]
  new gate
  gate.a = a
  gate.b = b
  result.grad = zeros_like result.value
  result.requires_grad = true
  register_node("SubBroadcast", gate, sub_broadcast_backward_ag[TT], result, a, b)

proc `-.`*[TT](a, b: Variable[TT]): Variable[TT] =
  when compileOption("boundChecks"):
    check_ctx(a, b)
  new result
  result.context = a.context
  result.value = a.value -. b.value
  if a.is_grad_needed or b.is_grad_needed:
    result.sub_broadcast_cache(a, b)

# Variable + Tensor (e.g. adding a constant attention mask)

proc `+.`*[TT](a: Variable[TT], b: TT): Variable[TT] =
  a +. a.context.variable(b)

proc `+.`*[TT](b: TT, a: Variable[TT]): Variable[TT] =
  a +. b

