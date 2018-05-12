# Copyright (c) 2018 the Arraymancer contributors
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.


import ../../src/arraymancer
import unittest

suite "[NN Primitives - Gated Recurrent Unit]":
  let x = toTensor([[ 0.1,  0.2,  0.3,  0.4],
                    [-0.1, -0.2, -0.3, -0.4],
                    [ 0.5,  0.6,  0.7,  0.8]])

  let hidden = toTensor([
    [ -1.0, -1.0],
    [ -1.0, -1.0],
    [ -1.0, -1.0]])

  let W3 = toTensor([
    [0.9, 0.8, 0.7, 0.6],
    [0.8, 0.7, 0.6, 0.5],
    [0.7, 0.6, 0.5, 0.4],
    [0.6, 0.5, 0.4, 0.3],
    [0.5, 0.4, 0.3, 0.2],
    [0.4, 0.3, 0.2, 0.1]])

  let U3 = toTensor([
    [-0.3, -0.1],
    [-0.2,  0.0],
    [-0.3, -0.1],
    [-0.2,  0.0],
    [-0.3, -0.1],
    [-0.2,  0.0],
  ])

  let bW3 = toTensor(
    [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]]
    )

  let bU3 = toTensor(
    [[-0.1, -0.2, -0.3, -0.4, -0.5, -0.6]],
    )

  test "GRU Cell inference only - equivalent to PyTorch/CuDNN":

    var hprime: Tensor[float64]

    gru_cell_inference(x, hidden,
                    W3, U3,
                    bW3, bU3,
                    h_prime)

    let pytorch_expected = [[-0.5317, -0.4753],
                            [-0.3930, -0.3210],
                            [-0.7325, -0.6430]].toTensor

    check: mean_relative_error(pytorch_expected, hprime) < 1e-4

  test "GRU Cell forward - equivalent to PyTorch/CuDNN":

    var r, z, n, Uh = randomTensor[float64](
      hidden.shape[0], hidden.shape[1], Inf # Making sure that values are overwritten
      )

    var hprime: Tensor[float64]

    gru_cell_forward(x, hidden,
                    W3, U3,
                    bW3, bU3,
                    r, z, n, Uh,
                    h_prime)

    let pytorch_expected = [[-0.5317, -0.4753],
                            [-0.3930, -0.3210],
                            [-0.7325, -0.6430]].toTensor

    check: mean_relative_error(pytorch_expected, hprime) < 1e-4
