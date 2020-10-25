==============================
Spellbook: How to do a multilayer perceptron
==============================

.. code:: nim

    import arraymancer

    # Learning XOR function with a neural network.

    # Autograd context / neuralnet graph
    let ctx = newContext Tensor[float32]

    let bsz = 32 # batch size

    # We will create a tensor of size 3200 (100 batches of size 32)
    # We create it as int between [0, 2[ and convert to bool
    let x_train_bool = randomTensor([bsz * 100, 1], 2).astype(bool)

    # Let's build our truth labels. We need to apply xor between the 2 columns of the tensors
    let y_bool = x_train_bool[_,0] xor x_train_bool[_,1]

    # Convert to float
    let x_train = ctx.variable(x_train_bool.astype(float32))
    let y = y_bool.astype(float32)

    # We will build the following network:
    # Input --> Linear(out_features = 3) --> relu --> Linear(out_features = 1) --> Sigmoid --> Cross-Entropy Loss

    # First hidden layer of 3 neurons, shape [3 out_features, 2 in_features]
    # We initialize with random weights between -1 and 1
    let layer_3neurons = ctx.variable(
                          randomTensor(3, 2, 2.0f) -. 1.0f
                          )

    # Classifier layer with 1 neuron per feature. (In our case only one neuron overall)
    # We initialize with random weights between -1 and 1
    let classifier_layer = ctx.variable(
                      randomTensor(1, 3, 2.0f) -. 1.0f
                      )

    # Stochastic Gradient Descent
    let optim = newSGD[float32](
      layer_3neurons, classifier_layer, 0.01f # 0.01 is the learning rate
    )

    # Learning loop
    for epoch in 0..5:
      for batch_id in 0..<100:

        # minibatch offset in the Tensor
        let offset = batch_id * 32
        let x = x_train[offset ..< offset + 32, _]
        let target = y[offset ..< offset + 32, _]

        # Building the network
        let n1 = relu linear(x, layer_3neurons)
        let n2 = linear(n1, classifier_layer)
        let loss = n2.sigmoid_cross_entropy(target)

        echo "Epoch is:" & $epoch
        echo "Batch id:" & $batch_id
        echo "Loss is:" & $loss.value

        # Compute the gradient (i.e. contribution of each parameter to the loss)
        loss.backprop()

        # Correct the weights now that we have the gradient information
        optim.update()
