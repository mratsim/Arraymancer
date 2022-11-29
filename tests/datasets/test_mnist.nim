# Copyright (c) 2018 the Arraymancer contributors
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).

import ../../src/arraymancer
import unittest

proc main() =
  suite "Datasets - MNIST":
    test "Load MNIST":
      let mnist = load_mnist(cache = true)
      check:
        mnist.train_images.shape == [60000, 28, 28]
        mnist.test_images.shape == [10000, 28, 28]
        mnist.train_labels.shape == [60000]
        mnist.test_labels.shape == [10000]

main()
GC_fullCollect()
