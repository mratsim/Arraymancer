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

type EmbeddingGate*[TT; scaled: static bool; Idx: SomeNumber or byte or char or enum] {.final.} = ref object of Gate[TT]
  cached_input_vocab_id: Tensor[Idx]
  weight: Variable[TT]
  padding_idx: Idx
    # We special-case -1 to mean no padding. Ideally we should use an option,
    # and have a separate proc for padding and no padding (to avoid costly checks within a tight loop)

proc embedding_backward_ag[TT; scaled: static bool, Idx](
        self: EmbeddingGate[TT, scaled, Idx],
        payload: Payload[TT]): SmallDiffs[TT] {.noInit.}=
  result = newDiffs[TT](1)
  result[0] = zeros_like(self.weight.value)
  let gradOutput = payload.variable.grad
  if self.weight.requires_grad:
    embedding_backward(result[0], self.cached_input_vocab_id, gradOutput, self.padding_idx, scaled)

proc embedding_cache[TT, Idx](
      result: Variable[TT],
      input_vocab_id: Tensor[Idx],
      weight: Variable[TT],
      padding_idx: Idx,
      scale_grad_by_freq: static[bool]
    ) =

  # Gate
  var gate: EmbeddingGate[TT, scale_grad_by_freq, Idx]
  new gate
  gate.cached_input_vocab_id = input_vocab_id
  gate.weight = weight
  gate.padding_idx = padding_idx

  # Result setup
  result.grad = zeros_like(result.value)
  result.requires_grad = true

  # Add to graph
  register_node(
    "Embedding",
    gate,
    # Why do we not need to cast to Backward[TT]
    # while we need to in sparse_softmax_cross_entropy?
    embedding_backward_ag[TT, scale_grad_by_freq, Idx],
    result,
    weight
  )


proc embedding*[TT; Idx: byte or char or SomeInteger](
        input_vocab_id: Tensor[Idx],
        weight: Variable[TT],
        padding_idx: Idx = -1,
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

  # Resulting var
  new result
  result.context = weight.context
  result.value = embedding(input_vocab_id, weight.value)

  # Caching for backprop
  if weight.is_grad_needed:
    embedding_cache(
      result,
      input_vocab_id, weight,
      padding_idx, scale_grad_by_freq
    )
