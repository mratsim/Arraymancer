# Copyright 2017-2018 Mamy André-Ratsimbazafy & the Arraymancer contributors
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

import  ../../private/sequninit,
        ../../tensor/tensor,
        ../../autograd/autograd,
        ../../nn_primitives/nn_primitives

type GRUGate*{.final.}[Layers, Timesteps: static[int], TT] = ref object of Gate[TT]
  ## For now the GRU layer only supports fixed size GRU stack and Timesteps
  cached_inputs: array[Layers, TT]
  cached_hiddens: array[Layers, array[Timesteps, TT]]
  W3s0, W3sN: Variable[TT]      # Weights
  U3s, bW3s, bU3s: Variable[TT] # Weights
  rs, zs, ns, Uhs: TT           # Intermediate tensors for backprop
  # TODO: store hidden_state for weight sharing?

proc forward[Layers, Timesteps: static[int], TT](
          self: GRUGate[Layers, TimeSteps, TT],
          a: Variable[TT], hidden0: Variable[TT],
          ): tuple[output, hiddenN: Variable[TT]] =
  ## Hidden_state is update in-place it's both an input and output

  new result.output
  new result.hiddenN

  result.output.context = a.context
  result.hiddenN.context = a.context
  if not a.is_grad_needed:
    gru_inference(
      a.data,
      self.W3s0.data, self.W3sN.data, self.U3s.data,
      self.bW3s.data, self.bU3s.data, result.data, hidden0.data
      )
  else:
    gru_forward(
      a.data, self.W3s0.data, self.W3sN.data, self.U3s.data,
      self.bW3s.data, self.bU3s.data,
      self.rs, self.zs, self.ns, self.Uhs,
      result.data, hidden0.data,
      self.cached_inputs,
      self.cached_hiddens
    )

method backward*[Layers, Timesteps: static[int], TT](
          self: GRUGate[Layers, TimeSteps, TT],
          payload: Payload[TT],
          ): SmallDiffs[TT] {.noInit.}=
  let gradient = payload.variable.grad
  result = newSeqUninit[TT](7)
  gru_backward(
    result[0], result[1],            # dinput, dhidden,
    result[2], result[3],            # dW3s0, dW3sN,
    result[4], result[5], result[6], # dU3s, dbW3s, dbU3s
    gradient,
    self.cached_inputs,
    self.cached_hiddens,
    self.W3s0, self.W3sN, self.U3s,
    self.rs, self.zs, self.ns, self.Uhs
  )

proc gru*[TT](
      input, hidden0: Variable[TT], Layers, Timesteps: static[int],
      W3s0, W3sN, U3s: Variable[TT],
      bW3s, bU3s: Variable[TT]
      ): tuple[output, hiddenN: Variable[TT]] =
  ## ⚠️ : Only compile-time known number of layers and timesteps is supported at the moment.
  ##      Bias cannot be nil at the moment
  ## Input:
  ##     - ``input`` Variable wrapping a 3D Tensor of shape [sequence/timesteps, batch, features]
  ##     - ``Layers`` A compile-time constant int corresponding to the number of stacker GRU layers
  ##     - ``Timesteps`` A compile-time constant int corresponding to the number of stacker GRU layers
  ##     - ``W3s0`` and ``W3sN`` Size2D tuple with height and width of the padding
  ##     - ``stride`` Size2D tuple with height and width of the stride
  ##
  ## Returns:
  ##     - A variable with a convolved 4D Tensor of size [N,C_out,H_out,W_out], where
  ##        H_out = (H_in + (2*padding.height) - kH) / stride.height + 1
  ##        W_out = (W_in + (2*padding.width) - kW) / stride.width + 1

  # TODO bound checking

  # Gate
  var gate: GRUGate[Layers, Timesteps, TT]
  new gate
  gate.W3s0 = W3s0
  gate.W3sN = W3sN
  gate.U3s = U3s
  gate.bW3s = bW3s
  gate.bU3s = bU3s

  # Node
  var node: Node[TT]
  new node

  node.gate = gate
  node.parents = newSeqUninit[VariablePtr[TT]](7)
  node.parents[0] = input.weakRef
  node.parents[1] = hidden0.weakRef
  node.parents[2] = W3s0.weakRef
  node.parents[3] = W3sN.weakRef
  node.parents[4] = U3s.weakRef
  node.parents[5] = bW3s.weakRef
  node.parents[6] = bU3s.weakRef

  # Resulting Variable
  result = gate.forward(input, hidden0)

  # Since output == hidden we need one Node for each
  # with only the payload differing
  var node_hidden: Node[TT]
  node_hidden[] = node[]

  node.payload = Payload[TT](kind: pkVar, variable: result.output)
  node_hidden.payload = Payload[TT](kind: pkVar, variable: result.hiddenN)

  input.context.push(node)
  input.context.push(node_hidden)

  # Caching for backprop
  if input.is_grad_needed or hidden0.is_grad_needed or
      W3s0.is_grad_needed or W3sN.is_grad_needed or
      U3s.is_grad_needed or
      bW3s.is_grad_needed or bU3s.is_grad_needed:

    result.output.grad = zeros_like(result.output.value)
    result.output.requires_grad = true

    result.hiddenN.grad = zeros_like(result.hiddenN.value)
    result.hiddenN.requires_grad = true
