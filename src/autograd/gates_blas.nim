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
        ./ag_data_structure

type MatMulGate* {.final.} [TT] = ref object of Gate[TT]
  ## TODO: generalize to C <- alpha AB + C
  a: Variable[TT]
  b: Variable[TT]

method forward*[TT](self: MatMulGate[TT], a, b: Variable[TT]): Variable[TT] {.inline, locks:0.}=
  new result

  result.tape = a.tape
  result.value = a.value * b.value
  result.grad = zeros[getSubType(TT)](result.value.shape)

method backward*[TT](self: MatMulGate[TT], gradient: TT): SmallDiffs[TT] {.noInit, inline, locks:0.}=
  result[0] = gradient * self.b.value.transpose
  result[1] = self.a.value.transpose * gradient

proc `*`*[TT](a, b: Variable[TT]): Variable[TT] =
  when compileOption("boundChecks"):
    check_ctx(a, b)

  # Gate
  var gate: MatMulGate[TT]
  new gate
  gate.arity = 2
  gate.a = a # TODO use ref to avoid copy
  gate.b = b

  # Node
  var node: Node[TT]
  new node

  node.gate = gate
  node.parents[0] = a.weakRef
  node.parents[1] = b.weakRef

  a.tape.push(node)

  # Resulting var
  result = gate.forward(a, b)
  node.child = result