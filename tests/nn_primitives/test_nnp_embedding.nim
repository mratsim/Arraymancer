# Copyright (c) 2018 Mamy André-Ratsimbazafy & the Arraymancer contributors
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

import ../../src/arraymancer, ../testutils
import unittest, tables, sequtils, strutils

proc main() =
  suite "[NN Primitive] Embedding":
    ## Embedding matrix that maps a vocabulary
    ## of size 21 into a 5-dimensional space
    ## (initialization completely random of course)
    let embed_matrix = [
      [0.10, 0.20, 0.30, 0.40, 0.50], # winter
      [0.50, 0.60, 0.70, 0.80, 0.90], # is
      [0.90, 0.80, 0.70, 0.60, 0.50], # coming
      [0.05, 0.06, 0.07, 0.08, 0.09], # you
      [0.15, 0.16, 0.17, 0.18, 0.19], # can
      [0.25, 0.26, 0.27, 0.28, 0.29], # cut
      [0.35, 0.36, 0.37, 0.38, 0.39], # all
      [0.45, 0.46, 0.47, 0.48, 0.49], # the
      [0.55, 0.56, 0.57, 0.58, 0.59], # flowers
      [0.65, 0.66, 0.67, 0.68, 0.69], # but
                                      # (you)
      [0.85, 0.86, 0.87, 0.88, 0.89], # cannot
      [0.50, 0.40, 0.30, 0.20, 0.10], # keep
      [0.10, 0.30, 0.50, 0.70, 0.90], # spring
      [0.15, 0.25, 0.35, 0.45, 0.55], # from
                                      # (coming)
      [0.22, 0.33, 0.44, 0.55, 0.66], # in
      [0.33, 0.44, 0.55, 0.66, 0.77], # seed
      [0.44, 0.55, 0.66, 0.77, 0.88], # time
      [0.55, 0.66, 0.77, 0.88, 0.99], # learn
                                      # (in)
      [0.99, 0.88, 0.77, 0.66, 0.55], # harvest
      [0.88, 0.77, 0.66, 0.55, 0.44], # teach
                                      # (in)
                                      # (winter)
      [0.77, 0.66, 0.55, 0.44, 0.33], # enjoy
      [1.00, 1.00, 1.00, 1.00, 1.00], # <START>
      [0.50, 0.50, 0.50, 0.50, 0.50], # <STOP>
      [0.00, 0.00, 0.00, 0.00, 0.00]  # <PAD>
    ].toTensor

    let words_to_ix = {
      "winter": 0,
      "is": 1,
      "coming": 2,
      "you": 3,
      "can": 4,
      "cut": 5,
      "all": 6,
      "the": 7,
      "flowers": 8,
      "but": 9,
      "cannot": 10,
      "keep": 11,
      "spring": 12,
      "from": 13,
      "in": 14,
      "seed": 15,
      "time": 16,
      "learn": 17,
      "harvest": 18,
      "teach": 19,
      "enjoy": 20,
      "<START>": 21,
      "<STOP>": 22,
      "<PAD>": 23
    }.toTable


    test "Embedding forward, input [batch_size, word_id]":
      # We create a batch of just the first words of our quotes
      let vocab_id = ["winter",
                      "you",
                      "in"].mapIt(words_to_ix[it]).toTensor

      # Sanity check
      check: vocab_id == [0, 3, 14].toTensor

      # Embedding check
      check: embedding(vocab_id, embed_matrix) == [
        [0.10, 0.20, 0.30, 0.40, 0.50], # winter
        [0.05, 0.06, 0.07, 0.08, 0.09], # you
        [0.22, 0.33, 0.44, 0.55, 0.66]  # in
      ].toTensor

    test "Embedding forward, input [batch_size, seq_len] and [seq_len, batch_size]":
      let sent1 = "<START> winter is coming <STOP>".splitWhitespace()
      let sent2 = "<START> you can cut all the flowers but you cannot keep spring from coming <STOP>".splitWhitespace()
      let sent3 = "<START> in seed time learn in harvest teach in winter enjoy <STOP>".splitWhitespace()

      let seq_len = max(max(sent1.len, sent2.len), sent3.len)

      let input_raw = [
        sent1 & sequtils.repeat("<PAD>", seq_len - sent1.len),
        sent2 & sequtils.repeat("<PAD>", seq_len - sent2.len),
        sent3 & sequtils.repeat("<PAD>", seq_len - sent3.len),
      ].toTensor

      check: input_raw.shape == [3, seq_len] # batch_size, seq_len

      let input_idx = input_raw.map_inline(words_to_ix[x])
      check: input_idx.at(0, _) == toTensor(@[21, 0, 1, 2, 22] & repeat(23, seq_len - sent1.len))

      block: # batch_size, seq_len
        let embed = embedding(input_idx, embed_matrix)

        check: embed.at(0, _, _) == toTensor(@[
          [1.00, 1.00, 1.00, 1.00, 1.00], # <START>
          [0.10, 0.20, 0.30, 0.40, 0.50], # winter
          [0.50, 0.60, 0.70, 0.80, 0.90], # is
          [0.90, 0.80, 0.70, 0.60, 0.50], # coming
          [0.50, 0.50, 0.50, 0.50, 0.50]  # <STOP>
        ] & repeat([0.00, 0.00, 0.00, 0.00, 0.00], seq_len - sent1.len) # <PAD>
        )

      block: # seq_len, batch_size
        # TODO
        echo "       Next test: [seq_len, batch_size] skipped"

    test "Embedding backpropagation - vocabulary of shape [batch_size]":
      const
        BatchSize =   3
        NbWords   =  21
        EmbedSize =   5
      let
        vocab: Tensor[int] = randomTensor([BatchSize], NbWords-1)
        embed_matrix: Tensor[float64] = randomTensor([NbWords, EmbedSize], 1.0)

      proc embed(embed_matrix: Tensor[float64]): float64 =
        embedding(vocab, embed_matrix).sum()
      let target_grad_embed = embed_matrix.numerical_gradient(embed)

      ## Backward pass
      let dOutput = ones[float64](BatchSize, EmbedSize)
      var dWeight = zeros_like(embed_matrix)

      embedding_backward(
        dWeight, vocab,
        dOutput,
        padding_idx = -1, # For testing we assume no padding (all indices contribute)
        scale_grad_by_freq = false # We don't reduce contribution of common words
      )

      check: dWeight.mean_absolute_error(target_grad_embed) < 1e-8

    test "Embedding backpropagation - vocabulary of shape [batch_size, seq_len]":
      const
        BatchSize =  10
        SeqLen    =  12
        NbWords   =  42
        EmbedSize =  11
      let
        vocab: Tensor[int] = randomTensor([BatchSize, Seqlen], NbWords-1)
        embed_matrix: Tensor[float64] = randomTensor([NbWords, EmbedSize], 1.0)

      proc embed(embed_matrix: Tensor[float64]): float64 =
        embedding(vocab, embed_matrix).sum()
      let target_grad_embed = embed_matrix.numerical_gradient(embed)

      ## Backward pass
      let dOutput = ones[float64](BatchSize, SeqLen, EmbedSize)
      var dWeight = zeros_like(embed_matrix)

      embedding_backward(
        dWeight, vocab,
        dOutput,
        padding_idx = -1, # For testing we assume no padding (all indices contribute)
        scale_grad_by_freq = false # We don't reduce contribution of common words
      )

      check: dWeight.mean_absolute_error(target_grad_embed) < 1e-8


main()
GC_fullCollect()
