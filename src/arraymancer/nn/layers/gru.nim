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
        ../../tensor,
        ../../autograd,
        ../../nn_primitives,
        sequtils

type GRUGate*[TT]{.final.}= ref object of Gate[TT]
  ## For now the GRU layer only supports fixed size GRU stack and Timesteps
  cached_inputs: seq[TT]
  cached_hiddens: seq[seq[TT]]
  W3s0, W3sN: Variable[TT]      # Weights
  U3s, bW3s, bU3s: Variable[TT] # Weights
  rs, zs, ns, Uhs: TT           # Intermediate tensors for backprop
  # TODO: store hidden_state for weight sharing?

proc gru_inference[TT](
          result: var tuple[output, hiddenN: Variable[TT]],
          input, hidden0: Variable[TT],
          W3s0, W3sN, U3s: Variable[TT],
          bW3s, bU3s: Variable[TT]
        ) =
  gru_inference(
    input.value,
    W3s0.value, W3sN.value, U3s.value,
    bW3s.value, bU3s.value, result.output.value, result.hiddenN.value
  )

proc gru_backward_ag[TT](
          self: Gate[TT],
          payload: Payload[TT],
        ): SmallDiffs[TT] {.noInit.}=
  let self = GRUGate[TT](self)
  let gradients = payload.sequence
  result = newDiffs[TT](7)
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

proc gru_forward[TT](
          result: var tuple[output, hiddenN: Variable[TT]],
          input, hidden0: Variable[TT],
          W3s0, W3sN, U3s: Variable[TT],
          bW3s, bU3s: Variable[TT]
        ) =
  ## Hidden_state is update in-place it's both an input and output

  # Gate
  var gate: GRUGate[TT]
  new gate

  gate.W3s0 = W3s0
  gate.W3sN = W3sN
  gate.U3s = U3s
  gate.bW3s = bW3s
  gate.bU3s = bU3s

  let layers = hidden0.value.shape[0]
  let seq_len = input.value.shape[0]
  let batch_size = input.value.shape[1]
  let hidden_size = hidden0.value.shape[2]

  gate.cached_inputs = newSeqUninit[TT](layers)
  gate.cached_hiddens = newSeqWith(layers) do: newSeq[TT](seq_len)

  gate.rs = newTensorUninit[TT.T](layers, seq_len, batch_size, hidden_size)
  gate.zs = newTensorUninit[TT.T](layers, seq_len, batch_size, hidden_size)
  gate.ns = newTensorUninit[TT.T](layers, seq_len, batch_size, hidden_size)
  gate.Uhs = newTensorUninit[TT.T](layers, seq_len, batch_size, hidden_size)

  # Compute
  gru_forward(
    input.value, W3s0.value, W3sN.value, U3s.value,
    bW3s.value, bU3s.value,
    gate.rs, gate.zs, gate.ns, gate.Uhs,
    result.output.value, result.hiddenN.value,
    gate.cached_inputs,
    gate.cached_hiddens
  )

  # Result setup
  result.output.grad = zeros_like(result.output.value)
  result.output.requires_grad = true

  result.hiddenN.grad = zeros_like(result.hiddenN.value)
  result.hiddenN.requires_grad = true

  # Add to graph
  register_node(
    "GRU",
    gate,
    gru_backward_ag[TT],
    @[result.output, result.hiddenN],
    input, hidden0,
    W3s0, W3sN, U3s,
    bW3s, bU3s
  )

proc gru*[TT](
      input, hidden0: Variable[TT],
      W3s0, W3sN, U3s: Variable[TT],
      bW3s, bU3s: Variable[TT]
      ): tuple[output, hiddenN: Variable[TT]] =
  ## ⚠️ : Only compile-time known number of layers and timesteps is supported at the moment.
  ##      Bias cannot be nil at the moment
  ## Input:
  ##     - ``input`` Variable wrapping a 3D Tensor of shape [sequence/timesteps, batch, features]
  ##     - ``hidden0`` the initial hidden state of shape [num_stacked_layers, batch, hidden_size]
  ##     - ``Timesteps`` A compile-time constant int corresponding to the number of stacker GRU layers
  ##     - ``W3s0`` and ``W3sN`` Size2D tuple with height and width of the padding
  ##     - ``stride`` Size2D tuple with height and width of the stride
  ##
  ## Outputs:
  ##   - `output` of shape [sequence/timesteps, batch, num_directions * hidden_size].
  ##     `output` contains the output features `hiddenT` for each T (timesteps)
  ##   - `hidden` of shape [num_stacked_layers * num_directions, batch, hidden_size].
  ##     `hidden` contains the hidden state for timestep T == sequence/timesteps length of `input`

  # Checks - TODO more checks
  doAssert hidden0.value.shape[1] == input.value.shape[1], "input batch_size: " & $input.value.shape[1] & " - hidden0 batch_size: " & $hidden0.value.shape[1]

  # initializing result
  new result.output
  new result.hiddenN
  result.output.context = input.context
  result.hiddenN.context = input.context
  result.hiddenN.value = hidden0.value.clone()

  # Training
  if input.is_grad_needed or hidden0.is_grad_needed or
      W3s0.is_grad_needed or W3sN.is_grad_needed or
      U3s.is_grad_needed or
      bW3s.is_grad_needed or bU3s.is_grad_needed:
    result.gru_forward(input, hidden0, W3s0, W3sN, U3s, bW3s, bU3s)
  else:
    result.gru_inference(input, hidden0, W3s0, W3sN, U3s, bW3s, bU3s)
