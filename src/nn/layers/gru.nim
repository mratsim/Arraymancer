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

import  ../../private/[ast_utils, sequninit],
        ../../tensor/tensor,
        ../../autograd/autograd,
        ../../nn_primitives/nn_primitives,
        sequtils

type GRUGate*{.final.}[TT] = ref object of Gate[TT]
  ## For now the GRU layer only supports fixed size GRU stack and Timesteps
  cached_inputs: seq[TT]
  cached_hiddens: seq[seq[TT]]
  W3s0, W3sN: Variable[TT]      # Weights
  U3s, bW3s, bU3s: Variable[TT] # Weights
  rs, zs, ns, Uhs: TT           # Intermediate tensors for backprop
  # TODO: store hidden_state for weight sharing?

proc forward[TT](
          self: GRUGate[TT],
          a: Variable[TT], hidden0: Variable[TT],
          ): tuple[output, hiddenN: Variable[TT]] =
  ## Hidden_state is update in-place it's both an input and output

  new result.output
  new result.hiddenN

  result.output.context = a.context
  result.hiddenN.context = a.context
  result.hiddenN.value = hidden0.value.clone()
  if ( # input.is_grad_needed or hidden0.is_grad_needed or
      # TODO improve inference shortcut - https://github.com/mratsim/Arraymancer/issues/301
      self.W3s0.is_grad_needed or self.W3sN.is_grad_needed or
      self.U3s.is_grad_needed or
      self.bW3s.is_grad_needed or self.bU3s.is_grad_needed):
    gru_forward(
      a.value, self.W3s0.value, self.W3sN.value, self.U3s.value,
      self.bW3s.value, self.bU3s.value,
      self.rs, self.zs, self.ns, self.Uhs,
      result.output.value, result.hiddenN.value,
      self.cached_inputs,
      self.cached_hiddens
    )
  else:
    gru_inference(
      a.value,
      self.W3s0.value, self.W3sN.value, self.U3s.value,
      self.bW3s.value, self.bU3s.value, result.output.value, result.hiddenN.value
      )

method backward*[TT](
          self: GRUGate[TT],
          payload: Payload[TT],
          ): SmallDiffs[TT] {.noInit.}=
  let gradients = payload.sequence
  result = newSeqUninit[TT](7)
  gru_backward(
    result[0], result[1],                 # Result   dinput, dhidden,
    result[2], result[3],                 # Result   dW3s0, dW3sN,
    result[4], result[5], result[6],      # Result   dU3s, dbW3s, dbU3s
    gradients[0].grad, gradients[1].grad, # Incoming dOutput, dHiddenN
    self.cached_inputs,
    self.cached_hiddens,
    self.W3s0.value, self.W3sN.value, self.U3s.value,
    self.rs, self.zs, self.ns, self.Uhs
  )

proc gru*[TT](
      input, hidden0: Variable[TT], layers: int,
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
  var gate: GRUGate[TT]
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
  input.context.push(node)

  # Checks
  let seq_len = input.value.shape[0]
  let batch_size = input.value.shape[1]
  let hidden_size = hidden0.value.shape[2]

  doAssert hidden0.value.shape[0] == layers # TODO bidirectional
  doAssert hidden0.value.shape[1] == batch_size, " - hidden0: " & $hidden0.value.shape[1] & ", batch_size: " & $batch_size

  # Resulting Variable
  gate.cached_inputs = newSeqUninit[TT](layers)
  gate.cached_hiddens = newSeqWith(layers) do: newSeq[TT](seq_len)

  type T = getSubtype TT
  gate.rs = newTensorUninit[T](layers, seq_len, batch_size, hidden_size)
  gate.zs = newTensorUninit[T](layers, seq_len, batch_size, hidden_size)
  gate.ns = newTensorUninit[T](layers, seq_len, batch_size, hidden_size)
  gate.Uhs = newTensorUninit[T](layers, seq_len, batch_size, hidden_size)

  result = gate.forward(input, hidden0)
  node.payload = Payload[TT](
    kind: pkSeq,
    sequence: @[result.output, result.hiddenN]
    )

  # Caching for backprop
  if input.is_grad_needed or hidden0.is_grad_needed or
      W3s0.is_grad_needed or W3sN.is_grad_needed or
      U3s.is_grad_needed or
      bW3s.is_grad_needed or bU3s.is_grad_needed:

    result.output.grad = zeros_like(result.output.value)
    result.output.requires_grad = true

    result.hiddenN.grad = zeros_like(result.hiddenN.value)
    result.hiddenN.requires_grad = true
