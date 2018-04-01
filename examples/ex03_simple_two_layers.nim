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
  x = ctx.variable(randomTensor[float32](N, D_in, 1'f32), requires_grad = true)
  y = randomTensor[float32](N, D_out, 1'f32)

# ##################################################################
# Define the model. API will be significantly improved in the future
let
  # We randomly initialize all weights and bias between [-0.5, 0.5]
  fc1_w = ctx.variable(randomTensor(H, D_in, 1'f32) .- 0.5'f32, requires_grad = true)
    # Fully connected 1: D_in -> hidden

  fc2_w = ctx.variable(randomTensor(D_out, H, 1'f32) .- 0.5'f32, requires_grad = true)
    # Fully connected 2: hidden -> D_out

proc model[TT](x: Variable[TT]): Variable[TT] =
  x
    .linear(fc1_w)
    .relu
    .linear(fc2_w)

# ##################################################################
# Training

let optim = newSGD[float32](fc1_w, fc2_w, 1e-4'f32)

for t in 0 ..< 500:
  let
    y_pred = model(x)
    loss = mse_loss(y_pred, y)

  echo &"Epoch {t}: loss {loss.value[0]}"

  optim.zeroGrads()
  loss.backprop()
  optim.update()
