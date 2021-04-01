import ../src/arraymancer

# Learning XOR function with a neural network.
proc main() =
  # Autograd context / neuralnet graph
  let ctx = newContext Tensor[float32]
  let bsz = 32 # batch size

  let x_train_bool = randomTensor([bsz * 100, 2], 1).astype(bool)
  let y_bool = x_train_bool[_,0] xor x_train_bool[_,1]
  let x_train = ctx.variable(x_train_bool.astype(float32))
  let y = y_bool.astype(float32)

  # We will build the following network:
  # Input --> Linear(out_features = 3) --> relu --> Linear(out_features = 1) --> Sigmoid --> Cross-Entropy Loss

  let layer_3neurons = ctx.variable(
                        randomTensor(3, 2, 2.0f) -. 1.0f,
                        requires_grad = true
                      )

  let classifier_layer = ctx.variable(
                          randomTensor(1, 3, 2.0f) -. 1.0f,
                          requires_grad = true
                        )

  # Stochastic Gradient Descent
  let optim = newSGD[float32](
      layer_3neurons, classifier_layer, 0.01f
    )

  # Learning loop
  for epoch in 0..10000:
    for batch_id in 0..<100:

      # minibatch offset in the Tensor
      let offset = batch_id * 32
      let x = x_train[offset ..< offset + 32, _]
      let target = y[offset ..< offset + 32, _]

      # Building the network
      let n1 = relu linear(x, layer_3neurons)
      let n2 = linear(n1, classifier_layer)
      let loss = n2.sigmoid_cross_entropy(target)

      # Compute the gradient (i.e. contribution of each parameter to the loss)
      loss.backprop()

      # Correct the weights now that we have the gradient information
      optim.update()

main()
