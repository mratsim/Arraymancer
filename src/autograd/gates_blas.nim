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

import ./autograd, ../arraymancer, ./utils

type MatMulGate* {.final.} [TT] = ref object of Gate[TT]
  ## TODO: generalize to C <- alpha AB + C
  a: TT
  b: TT

method forward*[TT](self: MatMulGate[TT], a, b: Variable[TT]): Variable[TT] {.inline.}=
  new result

  result.tape = a.tape
  result.value = a.value * b.value

  ## Unfortunately using broadcasts to save memory doesn't work
  # let z_shape = newSeqWith(result.value.rank, 1) # We create a shape of 1 dimension that we will expand with broadcast
  # let z = zeros[getSubType(TT)](z_shape)
  # result.grad = z.unsafeBroadcast(result.value.shape) # to save memory, we allocate as low as possible

  result.grad = zeros[getSubType(TT)](result.value.shape)

method backward*[TT](self: MatMulGate[TT], gradient: TT): SmallDiffs[TT] =
  result[0] = gradient * self.b.transpose
  result[1] = self.a.transpose * gradient

proc `*`*[TT](a, b: Variable[TT]): Variable[TT] =
  # when compileOption("boundChecks"):
  #   check_ctx(a, b)

  # Gate
  var gate: MatMulGate[TT]
  new gate
  gate.arity = 2
  gate.a = a.value # TODO use ref to avoid copy
  gate.b = b.value

  # Node
  var node: Node[TT]
  new node

  node.gate = gate
  node.parents[0] = a
  node.parents[1] = b

  a.tape.push(node)

  # Resulting var
  result = gate.forward(a, b)
  result.ancestor = node
  node.child = result