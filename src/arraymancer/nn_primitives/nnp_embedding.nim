# Copyright (c) 2018 Mamy André-Ratsimbazafy & the Arraymancer contributors
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

import
  ../tensor, tables, math

proc flatten_idx(t: Tensor): Tensor {.inline.}=
  t.reshape(t.size)

proc embedding*[T; Idx: byte or char or SomeInteger](
      vocab_id: Tensor[Idx],
      weight: Tensor[T]
    ): Tensor[T] =
  ## Returns embeddings from a `weight` embedding matrix and `vocab_id`
  ## to represent the part of the global vocabulary present.
  ##
  ## The main use-case is for natural language processing.
  ## Words (or characters or group of words) need to be encoded into arbitrary
  ## integers first that will be used to index the `weight` embedding matrix.
  ##
  ## During training, words that are related will get become close in some dimensions
  ## of the embedding.
  ##
  ## For example, if we want to encode a text containing 10000 different words
  ## into a 300-dimensional vector, we will require a [10000, 300] embedding matrix.
  ##
  ## Make sure to add an index to represent <UNKNOWN> words.
  ## (Words present during test that didn't exist in the training vocabulary)
  ##
  ## If working with variable-length sequences a <START>, <STOP> and <PAD> "words"
  ## are also useful
  ##
  ## In summary it's a lookup table that maps words to meanings
  ## in a high-dimensional space and that can be trained.
  ##
  ## Input:
  ##   - A tensor of vocabulary indices, either:
  ##       - [batch_size]
  ##       - [seq_len, batch_size]
  ##       - [batch_size, seq_len]
  ##   - A weight matrix that maps those indices to the embedding
  ##     of shape [vocabulary_size, embedding_size]
  ##
  ## Result:
  ##   - Depending on the input vocabulary:
  ##       - [batch_size, embedding_size]
  ##       - [seq_len, batch_size, embedding_size]
  ##       - [batch_size, seq_len, embedding_size]

  if vocab_id.rank == 1:
    return weight.index_select(0, vocab_id)

  let shape = vocab_id.shape & weight.shape[1]
  result = weight.index_select(0, vocab_id.flatten_idx).reshape(shape)

proc embedding_backward*[T; Idx: byte or char or SomeInteger](
      dWeight: var Tensor[T],
      vocab_id: Tensor[Idx],
      dOutput: Tensor[T],
      padding_idx: Idx,
      scale_grad_by_freq: static[bool] = false # scale by the inverse document frequency, i.e. divide by count in minibatch
    ) =

  doAssert vocab_id.size == dOutput.size div dOutput.shape[^1], "Besides the last dimension, vocab_id and the gradient flowing back must have the same shape"

  when scale_grad_by_freq:
    let count_size = nextPowerOfTwo dWeight.shape[0] # vocabulary size
    var counts {.global.} = initCountTable[int](count_size) # Optim: we reuse the buffer across minibatches

    counts.clear()
    for word_idx in vocab_id:
      counts.inc word_idx

  # We assume that dWeight is zero initialized with shape
  # [vocabulary_size, embedding_size] for us.
  let flat_vocab_id = vocab_id.flatten_idx()
  let flat_dOutput = dOutput.flatten_idx()

  for i, word_idx in enumerate(flat_vocab_id):
    if word_idx != padding_idx:
      var grad_curr_word = dWeight[int(word_idx), _]
      when scale_grad_by_freq:
        # For speed don't respect IEEE-754 and avoid
        # division in tight loop by multiplying by the inverse
        let idf = 1.T div counts[word_idx] # inverse document frequency
        grad_curr_word +.= flat_dOutput[i] * idf
      else:
        grad_curr_word +.= flat_dOutput[i]
