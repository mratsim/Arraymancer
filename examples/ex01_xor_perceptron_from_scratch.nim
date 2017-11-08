import ../src/arraymancer

# Example multilayer perceptron in Arraymancer.

# We will use as examples the OR function similar to this article:
# https://blog.dbrgn.ch/2013/3/26/perceptrons-in-python/


# Okay let's start
# With x and y being one sample, the perceptron equation is
#
# Layer 1
# n1 = relu(a1 * x + b1 * y + c1) # First neuron + relu activation
# n2 = relu(a2 * x + b2 * y + c2) # 2nd neuron + relu activation
# n3 = relu(a3 * x + b3 * y + c3) # 3nd neuron + relu activation
#
# Layer 2
# classifier =  a4 * n1 + b4 * n2 + c4 * n3
#
# Loss
# loss = cross_entropy(sigmoid(classifier))

# In terms of high level layers this becomes:
# Input --> Linear(out_features = 3) --> relu --> Linear(out_features = 1) --> Sigmoid --> Cross-Entropy Loss

# Let's go

# First create a context that will store backpropagation information
let ctx = newContext Tensor[float32]

# We will pass batches of 32 samples
let bsz = 32 #batch size

# We will create a tensor of size 3200 --> 100 batch sizes of 32
# We create it as int between [0, 2[ (2 excluded) and convert to bool
let x_train_bool = randomTensor([bsz * 100, 2], 2).astype(bool) # generate batch_size examples of (0,1) combination

# Let's check the first 32
echo x_train_bool[0..<32, _]
# Tensor of shape 32x2 of type "bool" on backend "Cpu"
# |true   false|
# |true   true|
# |false  false|
# |false  true|
# |false  false|
# |false  false|
# |false  false|
# ...

# Let's build or truth labels. We need to apply xor between the 2 columns of the tensors
let y_bool = x_train_bool[_,0] xor x_train_bool[_,1]


echo y_bool[0..<32, _]
# Tensor of shape 32x1 of type "bool" on backend "Cpu"
#         true|
#         false|
#         false|
#         true|
#         false|
#         false|
#         false|
#         true|
#         false|
#         ...

# Convert to float.
# Important: At the moment, Arraymancer expects batch size to be last
# so we transpose. In the future Arraymancer will be flexible.
let x_train = ctx.variable(x_train_bool.astype(float32).transpose)
let y = y_bool.astype(float32).transpose

# Now we create layer of neurons W that we will train to reproduce the xor function.
# Weights are of this shape: [W: out_features, in_features]

# First hidden layer of 3 neurons, with 2 features in
# We initialize with random weights between -1 and 1
# (We initialize them between 0.0f and 2.0f and then minus 1.0f)
# .- is the minus broadcasting operator
let layer_3neurons = ctx.variable(
                      randomTensor(3, 2, 2.0f) .- 1.0f
                      )

# Classifier layer with 1 neuron per feature. (In our case only one neuron overall)
# We initialize with random weights between -1 and 1
let classifier_layer = ctx.variable(
                  randomTensor(1, 3, 2.0f) .- 1.0f
                  )
# We use Stochastic Gradient Descent as optimizer
# With gradient descent the weigth are updated as follows:
# W -= learning_rate * dW
let optim = newSGD[float32](
  layer_3neurons, classifier_layer, 0.01f # 0.01 is the learning rate
)

# Now let's setup the training loops.
# First loop is passing the mini-batch, bacpropagating, updating the gradients.
# We do it until the whole x_train tensor has been passed through.
# This is one "epoch".

# Usually after each epoch we "validate" with a test set that the network was never trained on
# how the network generalized. In this example we won't go there to keep it short.

# We will do 5 epochs, passing the 32*100 minibatches
for epoch in 0..5:

  for batch_id in 0..<100:

    # offset in the Tensor (Remember, batch size is last)
    let offset = batch_id * 32
    let x = x_train[_, offset ..< offset + 32]
    let target = y[_, offset ..< offset + 32]

    # Building the network
    let n1 = linear(x, layer_3neurons)
    let n1_relu = n1.relu
    let n2 = linear(n1_relu, classifier_layer)
    let loss = sigmoid_cross_entropy(n2, target)

    echo "Epoch is:" & $epoch
    echo "Batch id:" & $batch_id

    echo "Loss is:" & $loss.value.data[0]

    # Compute the gradient (i.e. contribution of each parameter to the loss)
    loss.backprop()

    # Correct the weights now that we have the gradient information
    optim.update()