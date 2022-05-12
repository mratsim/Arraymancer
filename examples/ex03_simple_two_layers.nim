# Nim port of jcjohnson code: https://github.com/jcjohnson/pytorch-examples/blob/master/nn/two_layer_net_nn.py

import ../src/arraymancer, strformat

discard """
A fully-connected ReLU network with one hidden layer, trained to predict y from x
by minimizing squared Euclidean distance.
"""

# ##################################################################
# Environment variables

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
let (N, D_in, H, D_out) = (64, 1000, 100, 10)

# Create the autograd context that will hold the computational graph
let ctx = newContext Tensor[float32]

# Create random Tensors to hold inputs and outputs, and wrap them in Variables.
let
  x = ctx.variable(randomTensor[float32](N, D_in, 1'f32))
  y = randomTensor[float32](N, D_out, 1'f32)

# ##################################################################
# Define the model.

network TwoLayersNet:
  layers:
    fc1: Linear(D_in, H)
    fc2: Linear(H, D_out)
  forward x:
    x.fc1.relu.fc2

let
  model = ctx.init(TwoLayersNet)
  optim = model.optimizer(SGD, learning_rate = 1e-4'f32)

# ##################################################################
# Training

for t in 0 ..< 500:
  let
    y_pred = model.forward(x)
    loss = y_pred.mse_loss(y)

  echo &"Epoch {t}: loss {loss.value[0]}"

  loss.backprop()
  optim.update()
