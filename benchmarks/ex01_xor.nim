import ../src/arraymancer

let ctx = newContext Tensor[float32]

let bsz = 32 #batch size

# We will create a tensor of size 3200 --> 100 batch sizes of 32
# We create it as int between [0, 2[ (2 excluded) and convert to bool
let x_train_bool = randomTensor([bsz * 100, 2], 2).astype(bool)

# Let's build or truth labels. We need to apply xor between the 2 columns of the tensors
proc xor_alt[T](x,y: T): T =
  ## xor is builtin and cannot be passed to map as is
  x xor y

let y_bool = map2(x_train_bool[_,0], xor_alt, x_train_bool[_,1])


# Convert to float and transpose so batch_size is last
let x_train = ctx.variable(x_train_bool.astype(float32).transpose)
let y = y_bool.astype(float32).transpose

# First hidden layer of 3 neurons, with 2 features in
# We initialize with random weights between -1 and 1
let layer_3neurons = ctx.variable(
                      randomTensor(3, 2, 2.0f) .- 1.0f
                      )

# Classifier layer with 1 neuron per feature. (In our case only one neuron overall)
# We initialize with random weights between -1 and 1
let classifier_layer = ctx.variable(
                  randomTensor(1, 3, 2.0f) .- 1.0f
                  )

# Stochastic Gradient Descent
let optim = newSGD[float32](
  layer_3neurons, classifier_layer, 0.01f # 0.01 is the learning rate
)

for epoch in 0..10000:

  for batch_id in 0..<100:

    # offset in the Tensor (Remember, batch size is last)
    let offset = batch_id * 32
    let x = x_train[_, offset ..< offset + 32]
    let target = y[_, offset ..< offset + 32]

    # Building the network
    let n1 = linear(x, layer_3neurons)
    let n1_act = n1.relu
    let n2 = linear(n1_act, classifier_layer)
    let loss = sigmoid_cross_entropy(n2, target)

    # Compute the gradient (i.e. contribution of each parameter to the loss)
    loss.backprop()

    # Correct the weights now that we have the gradient information
    optim.update()