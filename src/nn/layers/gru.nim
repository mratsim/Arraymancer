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

import  ../../tensor/tensor,
        ../../autograd/autograd,
        ../../nn_primitives/nn_primitives

type GRUGate*{.final.}[Layers, Timesteps: static[int], TT] = ref object of Gate[TT]
  ## For now the GRU layer only supports fixed size GRU stack and Timesteps
  cached_inputs: array[Layers, TT]
  cached_hiddens: array[Layers, array[Timesteps, TT]]
  W3s: array[Layers, TT] # Weight
  U3s, bW3s, bU3s: TT    # Weights
  rs, zs, ns, Uhs: TT    # Intermediate tensors for backprop

method forward*[Layers, Timesteps: static[int], TT](
          self: GRUGate[Layers, TimeSteps, TT],
          a: Variable[TT], hidden_state: Variable[TT],
          ): Variable[TT] =
  ## Hidden_state is update in-place it's both an input and output

  new result

  result.context = a.context
  if not a.is_grad_needed:
    gru_inference(a.data, self.W3s, self.U3s, self.bW3s, self.bU3s, result.data, hidden_state.data)
  else:
    gru_forward(
      a.data, self.W3s, self.U3s, self.bW3s, self.bU3s,
      self.rs, self.zs, self.ns, self.Uhs,
      result.data, hidden_state.data,
      self.cached_inputs,
      self.cached_hiddens
    )
