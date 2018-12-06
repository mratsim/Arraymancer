# Copyright (c) 2018 the Arraymancer contributors
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).

import ../../src/arraymancer
import unittest

suite "Datasets - IMDB":
  test "Load IMDB":
    let imdb = load_imdb(cache = true)
    check:
      imdb.train_texts.shape == [12500]
      imdb.test_texts.shape == [12500]
      imdb.train_labels.shape == [12500]
      imdb.test_labels.shape == [12500]