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

import  ../tensor,
        ./autograd_common,
        ./private/p_broadcast

type MatMulGate*[TT] {.final.} = ref object of Gate[TT]
  ## TODO: generalize to C <- alpha AB + C
  a: Variable[TT]
  b: Variable[TT]

proc matmul_backward_ag[TT](self: Gate[TT], payload: Payload[TT]): SmallDiffs[TT] =
  let self = MatMulGate[TT](self)
  let gradient = payload.variable.grad
  result = newDiffs[TT](2)
  if self.a.requires_grad:
    let rawA = gradient * self.b.value.transpose2d
    result[0] = reduce_broadcast_dims(rawA, self.a.value.shape)
  if self.b.requires_grad:
    let rawB = self.a.value.transpose2d * gradient
    result[1] = reduce_broadcast_dims(rawB, self.b.value.shape)

proc matmul_cache[TT](result: Variable[TT], a, b: Variable[TT]) =
  # Gate
  var gate: MatMulGate[TT]
  new gate
  gate.a = a
  gate.b = b

  # Result setup
  result.grad = zeros_like result.value
  result.requires_grad = true

  # Add to graph
  register_node(
    "MatMul",
    gate,
    matmul_backward_ag[TT],
    result,
    a, b
  )

proc `*`*[TT](a, b: Variable[TT]): Variable[TT] =
  when compileOption("boundChecks"):
    check_ctx(a, b)

  new result
  result.context = a.context
  result.value = a.value * b.value

  if a.is_grad_needed or b.is_grad_needed:
    result.matmul_cache(a, b)

# ############################################################
#
#                      Scaling Gate
#
# ############################################################

type ScaleGate*[T] {.final.} = ref object of Gate[Tensor[T]]
  scalar: T

proc scale_backward_ag[T](self: Gate[Tensor[T]], payload: Payload[Tensor[T]]): SmallDiffs[Tensor[T]] =
  let self = ScaleGate[T](self)
  result = newDiffs[Tensor[T]](1)
  result[0] = payload.variable.grad *. self.scalar

proc scale_cache[T](result: Variable[Tensor[T]], a: Variable[Tensor[T]], scalar: T) =
  var gate: ScaleGate[T]
  new gate
  gate.scalar = scalar

  result.grad = zeros_like result.value
  result.requires_grad = true

  register_node(
    "Scale",
    gate,
    scale_backward_ag[T],
    result,
    a
  )

proc `*`*[T: SomeNumber, S: SomeNumber](a: Variable[Tensor[T]], scalar: S): Variable[Tensor[T]] =
  let s = scalar.T
  new result
  result.context = a.context
  result.value = a.value *. s
  if a.is_grad_needed:
    result.scale_cache(a, s)

proc `*`*[T: SomeNumber, S: SomeNumber](scalar: S, a: Variable[Tensor[T]]): Variable[Tensor[T]] =
  a * scalar

proc `/`*[T: SomeNumber, S: SomeNumber](a: Variable[Tensor[T]], scalar: S): Variable[Tensor[T]] =
  let s = scalar.T
  new result
  result.context = a.context
  result.value = a.value /. s
  if a.is_grad_needed:
    result.scale_cache(a, 1.T / s)
