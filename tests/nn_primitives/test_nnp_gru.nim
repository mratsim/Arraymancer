# Copyright (c) 2018 the Arraymancer contributors
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.


import ../../src/arraymancer
import unittest

suite "[NN Primitives - Gated Recurrent Unit]":
  test "GRU Cell - forward equivalent to PyTorch/CuDNN":
    let x = toTensor([[ 0.1,  0.2,  0.3,  0.4],
                      [-0.1, -0.2, -0.3, -0.4],
                      [ 0.5,  0.6,  0.7,  0.8]])

    let hidden = toTensor([
      [ -1.0, -1.0],
      [ -1.0, -1.0],
      [ -1.0, -1.0]])

    let w_input = toTensor([
      [0.9, 0.8, 0.7, 0.6],
      [0.8, 0.7, 0.6, 0.5],
      [0.7, 0.6, 0.5, 0.4],
      [0.6, 0.5, 0.4, 0.3],
      [0.5, 0.4, 0.3, 0.2],
      [0.4, 0.3, 0.2, 0.1]])

    let w_recur = toTensor([
      [-0.3, -0.1],
      [-0.2,  0.0],
      [-0.3, -0.1],
      [-0.2,  0.0],
      [-0.3, -0.1],
      [-0.2,  0.0],
    ])

    let b_input = toTensor(
      [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]]
      )

    let b_recur = toTensor(
      [[-0.1, -0.2, -0.3, -0.4, -0.5, -0.6]],
      )

    var hprime: Tensor[float64]

    gru_cell_forward(x, hidden,
                    w_input, w_recur,
                    b_input, b_recur,
                    h_prime)

    let pytorch_expected = [[-0.5317, -0.4753],
                            [-0.3930, -0.3210],
                            [-0.7325, -0.6430]].toTensor

    check: mean_relative_error(pytorch_expected, hprime) < 1e-4
