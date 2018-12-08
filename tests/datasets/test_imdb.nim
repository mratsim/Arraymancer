# Copyright (c) 2018 the Arraymancer contributors
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).

import ../../src/arraymancer
import unittest

suite "Datasets - IMDB":
  test "Load IMDB":
    let imdb = load_imdb(cache = true)

    template is_pos(x: int): bool =
      (7 <= x) and (x <= 10)

    template is_neg(x: int): bool =
      (1 <= x) and (x <= 4)

    check:
      imdb.train_texts.shape == [25000]
      imdb.test_texts.shape == [25000]
      imdb.train_labels.shape == [25000]
      imdb.test_labels.shape == [25000]

      # Check at boundaries that we correctly read data
      # positive: [0, 12499]
      imdb.train_texts[0] != ""
      imdb.train_texts[1] != ""
      imdb.train_texts[12498] != ""
      imdb.train_texts[12499] != ""

      imdb.train_labels[0].is_pos()
      imdb.train_labels[1].is_pos()
      imdb.train_labels[12498].is_pos()
      imdb.train_labels[12499].is_pos()

      # negative: [12500, 24999]
      imdb.train_texts[12500] != ""
      imdb.train_texts[12501] != ""
      imdb.train_texts[24998] != ""
      imdb.train_texts[24999] != ""

      imdb.train_labels[12500].is_neg()
      imdb.train_labels[12501].is_neg()
      imdb.train_labels[24998].is_neg()
      imdb.train_labels[24999].is_neg()

      # positive: [0, 12499]
      imdb.test_texts[0] != ""
      imdb.test_texts[1] != ""
      imdb.test_texts[12498] != ""
      imdb.test_texts[12499] != ""

      imdb.test_labels[0].is_pos()
      imdb.test_labels[1].is_pos()
      imdb.test_labels[12498].is_pos()
      imdb.test_labels[12499].is_pos()

      # negative: [12500, 24999]
      imdb.test_texts[12500] != ""
      imdb.test_texts[12501] != ""
      imdb.test_texts[24998] != ""
      imdb.test_texts[24999] != ""

      imdb.test_labels[12500].is_neg()
      imdb.test_labels[12501].is_neg()
      imdb.test_labels[24998].is_neg()
      imdb.test_labels[24999].is_neg()
