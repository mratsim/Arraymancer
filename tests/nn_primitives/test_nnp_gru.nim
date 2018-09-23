# Copyright (c) 2018 the Arraymancer contributors
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.


import ../../src/arraymancer
import unittest, sequtils, strformat

suite "[NN Primitives - Gated Recurrent Unit - Cell]":
  block:
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

    let pytorch_expected = [[-0.5317437280, -0.4752916969],
                            [-0.3930421219, -0.3209550879],
                            [-0.7325259335, -0.6429624731]].toTensor

    test "GRU Cell inference only - equivalent to PyTorch/CuDNN":
      var h = hidden.clone()

      gru_cell_inference(x,
                        W3, U3,
                        bW3, bU3,
                        h)

      check: mean_relative_error(pytorch_expected, h) < 1e-8

    test "GRU Cell forward - equivalent to PyTorch/CuDNN":

      var r, z, n, Uh = randomTensor[float64](
        hidden.shape[0], hidden.shape[1], Inf # Making sure that values are overwritten
        )

      var h = hidden.clone()

      gru_cell_forward(x,
                      W3, U3,
                      bW3, bU3,
                      r, z, n, Uh,
                      h)

      check: mean_relative_error(pytorch_expected, h) < 1e-8

  ################################

  test "GRU Cell backpropagation":

    const
      HiddenSize = 2
      BatchSize = 3
      Features  = 4

    let # input values
      x      = randomTensor([BatchSize, Features], 1.0)
      hidden = randomTensor([BatchSize, HiddenSize], 1.0)
      W3     = randomTensor([3*HiddenSize, Features], 1.0)
      U3     = randomTensor([3*HiddenSize, HiddenSize], 1.0)
      bW3    = randomTensor([1, 3*HiddenSize], 1.0)
      bU3    = randomTensor([1, 3*HiddenSize], 1.0)

    # Creating the closures
    # We use the inference mode here
    # And test with forward-backward proc
    proc gru_x(x: Tensor[float64]): float64 =
      var output = hidden.clone()
      gru_cell_inference(x,
                        W3, U3,
                        bW3, bU3,
                        output)
      result = output.sum
    proc gru_hidden(hidden: Tensor[float64]): float64 =
      var output = hidden.clone()
      gru_cell_inference(x,
                        W3, U3,
                        bW3, bU3,
                        output)
      result = output.sum
    proc gru_W3(W3: Tensor[float64]): float64 =
      var output = hidden.clone()
      gru_cell_inference(x,
                        W3, U3,
                        bW3, bU3,
                        output)
      result = output.sum
    proc gru_U3(U3: Tensor[float64]): float64 =
      var output = hidden.clone()
      gru_cell_inference(x,
                        W3, U3,
                        bW3, bU3,
                        output)
      result = output.sum
    proc gru_bW3(bW3: Tensor[float64]): float64 =
      var output = hidden.clone()
      gru_cell_inference(x,
                        W3, U3,
                        bW3, bU3,
                        output)
      result = output.sum
    proc gru_bU3(bU3: Tensor[float64]): float64 =
      var output = hidden.clone()
      gru_cell_inference(x,
                        W3, U3,
                        bW3, bU3,
                        output)
      result = output.sum

    let # Compute the numerical gradients
      target_grad_x      = x.numerical_gradient(gru_x)
      target_grad_hidden = hidden.numerical_gradient(gru_hidden)
      target_grad_W3     = W3.numerical_gradient(gru_W3)
      target_grad_U3     = U3.numerical_gradient(gru_U3)
      target_grad_bW3    = bW3.numerical_gradient(gru_bW3)
      target_grad_bU3    = bU3.numerical_gradient(gru_bU3)

    var # Forward outputs
      next_hidden = zeros_like(hidden)
      # Value saved for backprop
      r = zeros_like(hidden)
      z = zeros_like(hidden)
      n = zeros_like(hidden)
      Uh = zeros_like(hidden)

    # gradients
    let grad_next_hidden = ones_like(next_hidden)
    var dx, dHidden, dW3, dU3, dbW3, dbU3: Tensor[float64]

    # Do a forward and backward pass
    var h = hidden.clone()
    gru_cell_forward(x,
                    W3, U3,
                    bW3, bU3,
                    r, z, n, Uh,
                    h)

    gru_cell_backward(dx, dHidden, dW3, dU3,
                      dbW3, dbU3,
                      grad_next_hidden,
                      x, hidden, W3, U3,
                      r, z, n, Uh)

    check:
      mean_relative_error(target_grad_x, dx) < 1e-4
      mean_relative_error(target_grad_hidden, dhidden) < 1e-4
      mean_relative_error(target_grad_W3, dW3) < 1e-4
      mean_relative_error(target_grad_U3, dU3) < 1e-4
      mean_relative_error(target_grad_bW3, dbW3) < 1e-4
      mean_relative_error(target_grad_bU3, dbU3) < 1e-4

suite "[NN Primitives - GRU: Stacked, sequences, bidirectional]":

  const
    Timesteps = 4
    Layers = 2

  let x = toTensor([
      [ [ 0.1,  0.2,  0.3,  0.4,  0.5],  # Sequence/timestep 1
        [-0.1, -0.2, -0.3, -0.4, -0.5],
        [ 0.5,  0.6,  0.7,  0.8,  0.9]],
      [ [-0.1, -0.2, -0.3, -0.4, -0.5],  # Sequence/timestep 2
        [-0.1, -0.2, -0.3, -0.4, -0.5],
        [ 0.5,  0.6,  0.7,  0.8,  0.9]],
      [ [ 0.1,  0.2,  0.3,  0.4,  0.5],  # Sequence/timestep 3
        [-0.1, -0.2, -0.3, -0.4, -0.5],
        [-0.1, -0.2, -0.3, -0.4, -0.5]],
      [ [-0.1, -0.2, -0.3, -0.4, -0.5],  # Sequence/timestep 4
        [-0.1, -0.2, -0.3, -0.4, -0.5],
        [-0.1, -0.2, -0.3, -0.4, -0.5]]
    ])

  let hidden = toTensor([
      [ [ -1.0, -1.0],  # Stacked layer 1
        [ -1.0, -1.0],
        [ -1.0, -1.0]],
      [ [  2.0,  3.0],  # Stacked layer 2
        [  2.0,  3.0],
        [  2.0,  3.0]]
    ])

  let W3s = [
      toTensor(
        [ [0.9, 0.8, 0.7, 0.6, 0.5], # Stacked layer 1
          [0.8, 0.7, 0.6, 0.5, 0.4],
          [0.7, 0.6, 0.5, 0.4, 0.3],
          [0.6, 0.5, 0.4, 0.3, 0.2],
          [0.5, 0.4, 0.3, 0.2, 0.1],
          [0.4, 0.3, 0.2, 0.1, 0.0]]),
      toTensor(
        [ [0.5, 0.6], # Stacked layer 2
          [0.4, 0.5],
          [0.3, 0.4],
          [0.2, 0.3],
          [0.1, 0.2],
          [0.0, 0.1]])
    ]

  let U3s = toTensor([
      [ [-0.3, -0.1], # Stacked layer 1
        [-0.2,  0.0],
        [-0.3, -0.1],
        [-0.2,  0.0],
        [-0.3, -0.1],
        [-0.2,  0.0]],
      [ [-0.4, -0.5], # Stacked layer 2
        [-0.4,  0.5],
        [-0.4, -0.5],
        [-0.4,  0.5],
        [-0.4, -0.5],
        [-0.4,  0.5]]
    ])

  let bW3s = toTensor([
      [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]], # Stacked layer 1
      [[0.2, 0.3, 0.4, 0.2, 0.3, 0.4]]  # Stacked layer 2
    ])

  let bU3s = toTensor([
      [[-0.1, -0.2, -0.3, -0.4, -0.5, -0.6]], # Stacked layer 1
      [[ 0.1,  0.2,  0.3,  0.4,  0.5,  0.6]], # Stacked layer 2
    ])

  let py_expected_output = [[[0.2564464804, 2.4324064858],
                            [0.3212793315, 2.4821776260],
                            [0.1983539289, 2.3849941275]],

                            [[0.1323593874, 2.1939630024],
                            [0.1591080583, 2.2447442148],
                            [0.0683269257, 2.1121420346]],

                            [[0.1233993158, 1.9977140846],
                            [0.1320258443, 2.0434525564],
                            [0.0904040251, 1.9111566097]],

                            [[0.1355803839, 1.8195602154],
                            [0.1369919053, 1.8612790359],
                            [0.1230976350, 1.7381913793]]].toTensor

  let py_expected_hiddenN = [[[0.0155439572, 0.1680427130],
                            [-0.0038861287, 0.1854366532],
                            [-0.0549487103, 0.1189245136]],

                            [[0.1355803839, 1.8195602154],
                            [0.1369919053, 1.8612790359],
                            [0.1230976350, 1.7381913793]]].toTensor

  test "GRU inference only - equivalent to PyTorch/CuDNN":

    var output: Tensor[float64]
    var h = hidden.clone()

    gru_inference(x,
                  W3s,
                  U3s, bW3s, bU3s,
                  output, h)

    check:
      mean_relative_error(py_expected_output, output) < 1e-8
      mean_relative_error(py_expected_hiddenN, h) < 1e-8

  test "GRU forward - equivalent to PyTorch/CuDNN":
    # 1 direction hence `hidden.shape[0] * 1`
    var
      rs, zs, ns, Uhs = randomTensor[float64](
          [hidden.shape[0] * 1, hidden.shape[1], hidden.shape[2]], Inf # Making sure that values are overwritten
        )
      output: Tensor[float64]
      h = hidden.clone()
      cached_inputs: array[Layers, Tensor[float64]]
      cached_hidden: array[Layers, array[Timesteps, Tensor[float64]]]

    gru_forward(x,
                W3s,
                U3s, bW3s, bU3s,
                rs, zs, ns, Uhs,
                output, h,
                cached_inputs, cached_hidden)

    check:
      mean_relative_error(py_expected_output, output) < 1e-8
      mean_relative_error(py_expected_hiddenN, h) < 1e-8

  func GRU_backprop(Layers, TimeSteps: static[int], tol = 1e-4) =
    let test_name = &"GRU backpropagation: {Layers} layer(s), {TimeSteps} timestep(s)"
    test test_name:
      const
        HiddenSize = 2
        BatchSize = 3
        Features = 4

      let # input values
        x       = randomTensor([TimeSteps, BatchSize, Features], 1.0)
        hidden  = randomTensor([Layers, BatchSize, HiddenSize], 1.0)
        W3s     = block:
                    var ret: array[Layers, Tensor[float64]]
                    ret[0] = randomTensor([3*HiddenSize, Features], 1.0)
                    for i in 1 ..< Layers:
                      ret[i] = randomTensor([3*HiddenSize, HiddenSize], 1.0)
                    ret
        U3s     = randomTensor([Layers, 3*HiddenSize, HiddenSize], 1.0)
        bW3s    = randomTensor([Layers, 1, 3*HiddenSize], 1.0)
        bU3s    = randomTensor([Layers, 1, 3*HiddenSize], 1.0)

      # Creating the closure for numerical gradient testing
      # We use inference mode here
      # and test with forward-backward procs
      proc gru_x(x: Tensor[float64]): float64 =
        var output: Tensor[float64]
        var h = hidden.clone()
        gru_inference(x,
                      W3s, U3s,
                      bW3s, bU3s,
                      output, h)
        result = output.sum
      proc gru_hidden(hidden: Tensor[float64]): float64 =
        var output: Tensor[float64]
        var h = hidden.clone()
        gru_inference(x,
                      W3s, U3s,
                      bW3s, bU3s,
                      output, h)
        result = output.sum

      # TODO: Numerical gradient doesn't work on seq of tensors :/
      # proc gru_W3(W3: openarray[Tensor[float64]]): float64 =
      #   var output: Tensor[float64]
      #   var h = hidden.clone()
      #   gru_inference(x,
      #                 W3s, U3s,
      #                 bW3s, bU3s,
      #                 output, h)
      #   result = output.sum
      proc gru_U3(U3: Tensor[float64]): float64 =
        var output: Tensor[float64]
        var h = hidden.clone()
        gru_inference(x,
                      W3s, U3s,
                      bW3s, bU3s,
                      output, h)
        result = output.sum
      proc gru_bW3(bW3: Tensor[float64]): float64 =
        var output: Tensor[float64]
        var h = hidden.clone()
        gru_inference(x,
                      W3s, U3s,
                      bW3s, bU3s,
                      output, h)
        result = output.sum
      proc gru_bU3(bU3: Tensor[float64]): float64 =
        var output: Tensor[float64]
        var h = hidden.clone()
        gru_inference(x,
                      W3s, U3s,
                      bW3s, bU3s,
                      output, h)
        result = output.sum

      let # Compute the numerical gradients
        target_grad_x      = x.numerical_gradient(gru_x)
        target_grad_hidden = hidden.numerical_gradient(gru_hidden)
        # target_grad_W3s     = W3s.numerical_gradient(gru_W3) # TODO numerical gradient doesn't work on arrays of tensors
        target_grad_U3s     = U3s.numerical_gradient(gru_U3)
        target_grad_bW3s    = bW3s.numerical_gradient(gru_bW3)
        target_grad_bU3s    = bU3s.numerical_gradient(gru_bU3)

      var # Forward outputs
        output: Tensor[float64]
        cached_inputs: array[Layers, Tensor[float64]]
        cached_hiddens: array[Layers, array[Timesteps, Tensor[float64]]]
        h = hidden.clone()
        rs = zeros_like(hidden)
        zs = zeros_like(hidden)
        ns = zeros_like(hidden)
        Uhs = zeros_like(hidden)

      # Gradients
      let grad_output = ones[float64]([TimeSteps, BatchSize, HiddenSize])
      var dx, dHidden, dU3s, dbW3s, dbU3s: Tensor[float64]
      var dW3s: array[Layers, Tensor[float64]]

      # Do a forward and backward pass
      gru_forward(x,
                  W3s, U3s,
                  bW3s, bU3s,
                  rs, zs, ns, Uhs,
                  output, h,
                  cached_inputs, cached_hiddens)

      gru_backward(dx, dHidden, dW3s, dU3s,
              dbW3s, dbU3s,
              grad_output,
              cached_inputs, cached_hiddens,
              W3s, U3s,
              rs, zs, ns, Uhs)

      check:
        mean_relative_error(target_grad_x, dx) < tol
        mean_relative_error(target_grad_hidden, dhidden) < tol
        # mean_relative_error(target_grad_W3s, dW3s) < tol # TODO numerical gradient doesn't work on arrays of tensors
        mean_relative_error(target_grad_U3s, dU3s) < tol
        mean_relative_error(target_grad_bW3s, dbW3s) < tol
        mean_relative_error(target_grad_bU3s, dbU3s) < tol

  GRU_backprop(4, 1, 1e-3)
  # GRU_backprop(1, 2, 1e-3)
