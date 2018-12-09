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
        sequtils, options

# template genEmbeddingGate(Tname: untyped): untyped =
#   ## We have 2 kinds of embeddings.
#   ## One where gradients are scaled by frequencies
#   ## One here they are not.
#   ## Due to runtime dispatch

type EmbeddingGate*{.final.}[TT; scaled: static bool; Idx: byte or char or SomeNumber] = ref object of Gate[TT]
  cached_input_vocab_id: Tensor[Idx]
  weight: Variable[TT]
  padding_idx: int
    # We special-case -1 to mean no padding. Ideally we should use an option,
    # and have a separate proc for padding and no padding (to avoid costly checks within a tight loop)

proc forward[TT, scaled, Idx](self: EmbeddingGate[TT, scaled, Idx], input: Tensor[Idx]): Variable[TT] {.inline.} =
  ## TODO: at which level should we manage the context

  new result

  result.context = self.weight.context
  result.value = embedding(input, self.weight)

method backward*[TT; scaled: static bool, Idx](self: EmbeddingGate[TT, scaled, Idx], payload: Payload[TT]): SmallDiffs[TT] {.noInit, inline.}=

  result = newSeqUninit[TT](1)
  result[0] = zeros_like(self.weight)

  let gradOutput = payload.variable.grad

  if self.weight.requires_grad:
    embedding_backward(result[0], self.input_vocab_id, gradOutput, self.padding_idx, scaled)

proc embedding*[TT; Idx: byte or char or SomeNumber](
        input_vocab_id: Tensor[Idx],
        weight: Variable[TT],
        padding_idx = -1,
        scale_grad_by_freq: static[bool] = false
      ): Variable[TT] =
  ## Input:
  ##   - A tensor of vocabulary indices, either:
  ##       - [batch_size]
  ##       - [seq_len, batch_size]
  ##       - [batch_size, seq_len]
  ##     Vocabulary can be words, characters, series of words.
  ##     Each item in your vocabulary must be encoded into an unique integer
  ##     before being passed to the Embedding layer.
  ##   - A `weight` matrix that maps those indices to the embedding vector space
  ##     of shape [vocabulary_size, embedding_size].
  ##   - An optional `padding_idx` if an index corresponds to the absence of words (padding)
  ##     This is necessary to support variable-length sentences.
  ##     By default, the `padding_idx` is `-1`.
  ##   - An optional parameter to scale the gradient by the words inverse document frequency.
  ##     This divides the gradient of each words by their occurences in the minibatch.
  ##     This regularise variations in the weight of very frequent words.

  # Gate
  var gate: EmbeddingGate[TT, scale_grad_by_freq]
  new gate
  gate.cached_input_vocab_id = input_vocab_id
  gate.weight = weight
  gate.padding_idx = padding_idx

  # Node
  var node: Node:[TT]
  new node

  node.gate = gate
  node.parents = newSeqUninit[VariablePtr[TT]](1)
  node.parents[0] = weight.weakRef

  weight.context.push(node)

  # Resulting var
  result = gate.forward(input_vocab_id)
  node.payload = Payload[TT](kind: pkVar, variable: result)

  # Caching for backprop
  if weight.is_grad_needed:
    result.grad = zeros_like(result.value)
    result.requires_grad = true
