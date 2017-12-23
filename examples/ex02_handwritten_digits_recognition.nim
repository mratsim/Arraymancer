import ../src/arraymancer, random

# This is an early minimum viable example of handwritten digits recognition.
# It uses convolutional neural networks to achieve high accuracy.
#
# Data files (MNIST) can be downloaded here http://yann.lecun.com/exdb/mnist/
# and must be decompressed in "./bin/" (or change the path "bin/..." below)
#
# Note:
# In the future, model, weights and optimizer definition will be streamlined.

# Make the results reproducible by initializing a random seed
randomize(42)

let
  ctx = newContext Tensor[float32] # Autograd/neural network graph
  n = 32                           # Batch size

let
  # Training data is 60k 28x28 greyscale images from 0-255,
  # neural net prefers input rescaled to [0, 1] or [-1, 1]
  x_train = read_mnist_images("bin/train-images.idx3-ubyte").astype(float32) / 255'f32

  # Change shape from [N, H, W] to [N, C, H, W], with C = 1 (unsqueeze). Convolution expect 4d tensors
  # And store in the context to track operations applied and build a NN graph
  X_train = ctx.variable x_train.unsqueeze(1)

  # Labels are uint8, we must convert them to int
  y_train = read_mnist_labels("bin/train-labels.idx1-ubyte").astype(int)

  # Idem for testing data (10000 images)
  x_test = read_mnist_images("bin/t10k-images.idx3-ubyte").astype(float32) / 255'f32
  X_test = ctx.variable x_test.unsqueeze(1)
  y_test = read_mnist_labels("bin/t10k-labels.idx1-ubyte").astype(int)

# Config (API is not finished)
let
  # We randomly initialize all weights and bias between [-0.5, 0.5]
  # In the future requires_grad will be automatically set for neural network layers

  cv1_w = ctx.variable(
    randomTensor(20, 1, 5, 5, 1'f32) .- 0.5'f32,    # Weight of 1st convolution
    requires_grad = true
    )
  cv1_b = ctx.variable(
    randomTensor(20, 1, 1, 1'f32) .- 0.5'f32,       # Bias of 1st convolution
    requires_grad = true
    )

  cv2_w = ctx.variable(
    randomTensor(50, 20, 5, 5, 1'f32) .- 0.5'f32,   # Weight of 2nd convolution
    requires_grad = true
    )

  cv2_b = ctx.variable(
    randomTensor(50, 1, 1, 1'f32) .- 0.5'f32,       # Bias of 2nd convolution
    requires_grad = true
    )

  fc3 = ctx.variable(
    randomTensor(500, 800, 1'f32) .- 0.5'f32,       # Fully connected: 800 in, 500 ou
    requires_grad = true
    )

  classifier = ctx.variable(
    randomTensor(10, 500, 1'f32) .- 0.5'f32,        # Fully connected: 500 in, 10 classes out
    requires_grad = true
    )

proc model[TT](x: Variable[TT]): Variable[TT] =
  # The formula of the output size of convolutions and maxpools is:
  #   H_out = (H_in + (2*padding.height) - kernel.height) / stride.height + 1
  #   W_out = (W_in + (2*padding.width) - kernel.width) / stride.width + 1

  let cv1 = x.conv2d(cv1_w, cv1_b).relu()      # Conv1: [N, 1, 28, 28] --> [N, 20, 24, 24]     (kernel: 5, padding: 0, strides: 1)
  let mp1 = cv1.maxpool2D((2,2), (0,0), (2,2)) # Maxpool1: [N, 20, 24, 24] --> [N, 20, 12, 12] (kernel: 2, padding: 0, strides: 2)
  let cv2 = mp1.conv2d(cv2_w, cv2_b).relu()    # Conv2: [N, 20, 12, 12] --> [N, 50, 8, 8]      (kernel: 5, padding: 0, strides: 1)
  let mp2 = cv2.maxpool2D((2,2), (0,0), (2,2)) # Maxpool1: [N, 50, 8, 8] --> [N, 50, 4, 4]     (kernel: 2, padding: 0, strides: 2)

  let f = mp2.flatten                          # [N, 50, 4, 4] -> [N, 800]
  let hidden = f.linear(fc3).relu              # [N, 800]      -> [N, 500]

  result = hidden.linear(classifier)           # [N, 500]      -> [N, 10]

# Stochastic Gradient Descent (API will change)
let optim = newSGD[float32](
  cv1_w, cv1_b, cv2_w, cv2_b, fc3, classifier, 0.01f # 0.01 is the learning rate
)

# Learning loop
for epoch in 0 ..< 5:
  for batch_id in 0 ..< X_train.value.shape[0] div n: # some at the end may be missing, oh well ...
    # minibatch offset in the Tensor
    let offset = batch_id * n
    let x = X_train[offset ..< offset + n, _]
    let target = y_train[offset ..< offset + n]

    # Running through the network and computing loss
    let clf = x.model
    let loss = clf.sparse_softmax_cross_entropy(target)

    if batch_id mod 200 == 0:
      # Print status every 200 batches
      echo "Epoch is: " & $epoch
      echo "Batch id: " & $batch_id
      echo "Loss is:  " & $loss.value.data[0]

    # Compute the gradient (i.e. contribution of each parameter to the loss)
    loss.backprop()

    # Correct the weights now that we have the gradient information
    optim.update()

  # Validation (checking the accuracy/generalization of our model on unseen data)
  ctx.no_grad_mode:
    echo "\nEpoch #" & $epoch & " done. Testing accuracy"

    # To avoid using too much memory we will compute accuracy in 10 batches of 1000 images
    # instead of loading 10 000 images at once
    var score = 0.0
    var loss = 0.0
    for i in 0 ..< 10:
      let y_pred = X_test[i ..< i+1000, _].model.value.softmax.argmax(axis = 1).indices.squeeze
      score += accuracy_score(y_test[i ..< i+1000, _], y_pred)

      loss += X_test[i ..< i+1000, _].model.sparse_softmax_cross_entropy(y_test[i ..< i+1000, _]).value.data[0]
    score /= 10
    loss /= 10
    echo "Accuracy: " & $(score * 100) & "%"
    echo "Loss:     " & $loss
    echo "\n"


############# Output ############

# Epoch is: 0
# Batch id: 0
# Loss is:  132.9124755859375
# Epoch is: 0
# Batch id: 200
# Loss is:  2.301989078521729
# Epoch is: 0
# Batch id: 400
# Loss is:  1.155071973800659
# Epoch is: 0
# Batch id: 600
# Loss is:  1.043337464332581
# Epoch is: 0
# Batch id: 800
# Loss is:  0.58299720287323
# Epoch is: 0
# Batch id: 1000
# Loss is:  0.5417937040328979
# Epoch is: 0
# Batch id: 1200
# Loss is:  0.6955615282058716
# Epoch is: 0
# Batch id: 1400
# Loss is:  0.4742314517498016
# Epoch is: 0
# Batch id: 1600
# Loss is:  0.3307125866413116
# Epoch is: 0
# Batch id: 1800
# Loss is:  0.6455222368240356

# Epoch #0 done. Testing accuracy
# Accuracy: 83.24999999999999%
# Loss:     0.5828457295894622


# Epoch is: 1
# Batch id: 0
# Loss is:  0.5344035029411316
# Epoch is: 1
# Batch id: 200
# Loss is:  0.4455387890338898
# Epoch is: 1
# Batch id: 400
# Loss is:  0.1642555445432663
# Epoch is: 1
# Batch id: 600
# Loss is:  0.5191419124603271
# Epoch is: 1
# Batch id: 800
# Loss is:  0.2091695368289948
# Epoch is: 1
# Batch id: 1000
# Loss is:  0.2661008834838867
# Epoch is: 1
# Batch id: 1200
# Loss is:  0.405451238155365
# Epoch is: 1
# Batch id: 1400
# Loss is:  0.1397259384393692
# Epoch is: 1
# Batch id: 1600
# Loss is:  0.526863694190979
# Epoch is: 1
# Batch id: 1800
# Loss is:  0.5916416645050049

# Epoch #1 done. Testing accuracy
# Accuracy: 88.49000000000001%
# Loss:     0.3582650691270828