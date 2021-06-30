# Nim port of jcjohnson code: https://github.com/jcjohnson/pytorch-examples/blob/master/nn/two_layer_net_nn.py

import ../../src/arraymancer, strformat, ../testutils
import unittest, random

proc gcntest() =
  # Test a simple Graph convolutional network
  # Each node has 2 features and the GCN layer has 2 output channels
  let ( D_in, D_out) = (2, 2)

  let ctx = newContext Tensor[float32]

  # Create two nodes with 2 node features
  # Create adjacency matrix to represent graph topological structure
  let
    x = ctx.variable(randomTensor[float32](2, D_in, 1'f32))
    adj = ctx.variable([[0,0], [0,1]].toTensor().asType(float32))
    y = randomTensor[float32](2, D_out, 1'f32)


  network ctx, GCNNet:
    layers:
      fc1: GCN(D_in, D_out)
    forward adj, x:
      fc1(x, adj)

  let
    model = ctx.init(GCNNet)
    optim = model.optimizerSGD(learning_rate = 1e-4'f32)

  let
      y_pred = model.forward(x, adj)
      loss = y_pred.sigmoid_cross_entropy(y)

  loss.backprop()
  optim.update()
  check y_pred.value.shape == [2,2]

testSuite "End-to-End: GCN network test":
  test "Run single layer GCN":
    gcntest()