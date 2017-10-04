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

import ./autograd, ../arraymancer, ./utils, sequtils

type MeanGate* {.final.} [TT] = ref object of Gate[TT]
  ## TODO: generalize to C <- alpha AB + C
  a_shape: seq[int]

method forward*[TT](self: MeanGate[TT], a: Variable[TT]): Variable[TT] {.inline, locks:0.}=
  new result

  result.tape = a.tape
  result.value = [a.value.mean].toTensor

  result.grad = zeros[getSubType(TT)](1)


method backward*[TT](self: MeanGate[TT], gradient: TT): SmallDiffs[TT] {.inline, locks:0.}=
  result[0] = gradient / getSubType(TT)(self.a_shape.product) # Conversion to subtype T, oh Higher kinded-types ...

  let z_shape = newSeqWith(self.a_shape.len, 1) # We create a shape of 1 dimension that we will expand with broadcast
  result[0] = result[0].unsafeReshape(z_shape).unsafeBroadcast(self.a_shape)

proc mean*[TT](a: Variable[TT]): Variable[TT] =
  when compileOption("boundChecks"):
    check_ctx(a, b)

  # Gate
  var gate: MeanGate[TT]
  new gate
  gate.arity = 1
  gate.a_shape = a.value.shape # TODO use ref to avoid copy

  # Node
  var node: Node[TT]
  new node

  node.gate = gate
  node.parents[0] = a

  a.tape.push(node)

  # Resulting var
  result = gate.forward(a)
  result.ancestor = node
  node.child = result