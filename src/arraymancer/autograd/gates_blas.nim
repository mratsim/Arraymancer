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
        ./autograd_common

type MatMulGate*[TT] {.final.} = ref object of Gate[TT]
  ## TODO: generalize to C <- alpha AB + C
  a: Variable[TT]
  b: Variable[TT]

proc matmul_backward_ag[TT](self: MatMulGate[TT], payload: Payload[TT]): SmallDiffs[TT] =
  let gradient = payload.variable.grad
  result = newDiffs[TT](2)
  result[0] = gradient * self.b.value.transpose
  result[1] = self.a.value.transpose * gradient

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
