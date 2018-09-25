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

import  ../../tensor/tensor,
        ../../nn_primitives/nn_primitives,
        ../../autograd/autograd

type GruGate* {.final.} [TT] = ref object of Gate[TT]
  input, hidden, W3, U3, bW3, bU3: Variable[TT]
  dinput, dhidden, dW3, dU3, dbW3, dbU3: TT
  r, z, n, Uh: TT

method forward*[TT](self: GruGate[TT], input: Variable[TT]): Variable[TT] {.inline, locks:0.}=
  new result
  result.context = input.context
  # I'm not sure if we should make this a binary `forward` with `input` and `hidden`
  # or if using `self.hidden` is preferred.
  gru_cell_forward(
    input.value, self.hidden.value,
    self.W3.value, self.U3.value,
    self.bW3.value, self.bU3.value,
    self.r, self.z, self.n, self.Uh,
    next_hidden=result.value
  )

method backward*[TT](self: GruGate[TT], gradOutput: TT): SmallDiffs[TT] {.noInit, inline, locks:0.}=
  gru_cell_backward(
    self.dinput, self.dhidden,        # output gradient w.r.t. input, hidden
    self.dW3, self.dU3,               # output gradient w.r.t. weights
    self.dbW3, self.dbU3,             # output gradient w.r.t. biases
    self.input, self.hidden,          # input parameters saved from forward
    self.W3, self.U3,                 # input parameters saved from forward
    self.r, self.z, self.n, self.Uh   # intermediate tensors saved from forward
  )
  # propagate gradients to parents
  if self.input.requires_grad:
    result[0] = self.dinput
  if self.hidden.requires_grad:
    result[1] = self.dhidden
  if self.W3.requires_grad:
    result[2] = self.dW3
  if self.U3.requires_grad:
    result[3] = self.dU3
  if self.bW3.requires_grad:
    result[4] = self.dbW3
  if self.bU3.requires_grad:
    result[5] = self.dbU3

proc gru*[TT](input, hidden, W3, U3, bW3, bU3: Variable[TT]): Variable[TT] =

  # Gate
  var gate = GruGate[TT]()
  gate.nb_grads = 6
  gate.input = input
  gate.hidden = hidden
  gate.W3 = W3
  gate.U3 = U3
  gate.bW3 = bW3
  gate.bU3 = bU3

  # Node
  var node = Node[TT]()
  node.gate = gate
  node.parents[0] = input.weakRef
  node.parents[1] = hidden.weakRef
  node.parents[2] = W3.weakRef
  node.parents[3] = U3.weakRef
  node.parents[4] = bW3.weakRef
  node.parents[5] = bU3.weakRef

  input.context.push(node)

  # Resulting var
  result = gate.forward(input)
  node.payload = result

  # Caching for backprop
  let grad_needed = (
    input.is_grad_needed or
    hidden.is_grad_needed or
    W3.is_grad_needed or
    U3.is_grad_needed or
    bW3.is_grad_needed or
    bU3.is_grad_needed
  )
  if grad_needed:
    # prepare gradient tensors for backward step
    gate.dinput = input.value.zeros_like()
    gate.dhidden = hidden.value.zeros_like()
    gate.dW3 = W3.value.zeros_like()
    gate.dU3 = U3.value.zeros_like()
    gate.dbW3 = bW3.value.zeros_like()
    gate.dbU3 = bU3.value.zeros_like()

    result.grad = result.value.zeros_like()
    result.requires_grad = true
