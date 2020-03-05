# Copyright 2017-Present Mamy André-Ratsimbazafy & the Arraymancer contributors
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
        ./autograd_common

type HadamardGate*[TT]{.final.} = ref object of Gate[TT]
  a: Variable[TT]
  b: Variable[TT]

proc hadamard_backward_ag[TT](self: HadamardGate[TT], payload: Payload[TT]): SmallDiffs[TT] =
  let gradient = payload.variable.grad
  result = newSeq[TT](2)
  result[0] = gradient     *. self.b.value
  result[1] = self.a.value *. gradient

proc hadamard_cache[TT](result: Variable[TT], a, b: Variable[TT]) =
  # Gate
  var gate: HadamardGate[TT]
  new gate
  gate.a = a
  gate.b = b

  # Result setup
  result.grad = zeros_like result.value
  result.requires_grad = true

  # Add to graph
  register_node(
    "Hadamard Product",
    gate,
    hadamard_backward_ag[TT],
    result,
    a, b
  )

proc `*.`*[TT](a, b: Variable[TT]): Variable[TT] =
  when compileOption("boundChecks"):
    check_ctx(a, b)

  # Resulting var
  new result
  result.context = a.context
  result.value = a.value *. b.value

  # Caching for backprop
  if a.is_grad_needed or b.is_grad_needed:
    result.hadamard_cache(a, b)
