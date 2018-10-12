# Copyright (c) 2018 Mamy André-Ratsimbazafy & the Arraymancer contributors
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

import
  ../tensor/tensor

func flatten_idx(t: Tensor): Tensor {.inline.}=
  t.reshape(t.size)

func embedding*[T](
      vocab_id: Tensor[int],
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
