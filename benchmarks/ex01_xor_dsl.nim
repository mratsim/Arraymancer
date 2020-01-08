import ../src/arraymancer

# Learning XOR function with a neural network.

# Autograd context / neuralnet graph
let ctx = newContext Tensor[float32]
let bsz = 32 # batch size

let x_train_bool = randomTensor([bsz * 100, 2], 1).astype(bool)
let y_bool = x_train_bool[_,0] xor x_train_bool[_,1]
let x_train = ctx.variable(x_train_bool.astype(float32))
let y = y_bool.astype(float32)

# We will build the following network:
# Input --> Linear(out_features = 3) --> relu --> Linear(out_features = 1) --> Sigmoid --> Cross-Entropy Loss

network ctx, XorNet:
  layers:
    x:          Input([2])
    hidden:     Linear(2, 3)
    classifier: Linear(3, 1)
  forward x:
    x.hidden.relu.classifier

let model = ctx.init(XorNet)

# Stochastic Gradient Descent
let optim = model.optimizerSGD(learning_rate = 0.01'f32)

# Learning loop
for epoch in 0..10000:
  for batch_id in 0..<100:

    # minibatch offset in the Tensor
    let offset = batch_id * 32
    let x = x_train[offset ..< offset + 32, _]
    let target = y[offset ..< offset + 32, _]

    # Running input through the network
    let output = model.forward(x)

    # Computing the loss
    let loss = output.sigmoid_cross_entropy(target)

    # Compute the gradient (i.e. contribution of each parameter to the loss)
    loss.backprop()

    # Correct the weights now that we have the gradient information
    optim.update()
