import ../src/arraymancer

# This is an early minimum viable example of handwritten digits recognition.
# It uses convolutional neural networks to achieve high accuracy.
#
# Data files (MNIST) can be downloaded here http://yann.lecun.com/exdb/mnist/
# and must be decompressed in "./bin/" (or change the path "bin/..." below)
#
# Note:
# In the future, model, weights and optimizer definition will be streamlined.
# Also, currently this only works on Nim 0.17.2
# until I debug the new Nim allocator introduced in November 2017 in devel branch.

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

  # Idem for testing data
  x_test = read_mnist_images("bin/t10k-images.idx3-ubyte").astype(float32) / 255'f32
  X_test = ctx.variable x_test.unsqueeze(1)
  y_test = read_mnist_labels("bin/t10k-labels.idx1-ubyte").astype(int)

# Config
let
  # We randomly initialize all weights and bias between [-0.5, 0.5]

  cv1_w = ctx.variable randomTensor[float32](20, 1, 5, 5, 1'f32) .- 0.5'f32  # Weight of 1st convolution
  cv1_b = ctx.variable randomTensor[float32](20, 1, 1, 1'f32) .- 0.5'f32     # Bias of 1st convolution

  cv2_w = ctx.variable randomTensor[float32](50, 20, 5, 5, 1'f32) .- 0.5'f32 # Weight of 2nd convolution
  cv2_b = ctx.variable randomTensor[float32](50, 1, 1, 1'f32) .- 0.5'f32     # Bias of 2nd convolution

  fc3 = ctx.variable randomTensor[float32](500, 800, 1'f32) .- 0.5'f32       # Fully connected: 800 in, 500 out

  classifier = ctx.variable randomTensor[float32](10, 500, 1'f32) .- 0.5'f32 # Fully connected: 500 in, 10 classes out

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
for epoch in 0..5:
  for batch_id in 0 ..< X_train.value.shape[0] div n: # some at the end may be missing, oh well ...
    # minibatch offset in the Tensor
    let offset = batch_id * n
    let x = X_train[offset ..< offset + n, _]
    let target = y_train[offset ..< offset + n]

    # Running through the network and computing loss
    let clf = x.model
    let loss = clf.sparse_softmax_cross_entropy(target)

    echo "Epoch is:" & $epoch
    echo "Batch id:" & $batch_id
    echo "Loss is:" & $loss.value.data[0]

    # Compute the gradient (i.e. contribution of each parameter to the loss)
    loss.backprop()

    # Correct the weights now that we have the gradient information
    optim.update()

  echo "\nEpoch #" & $epoch & " done. Testing accuracy"
  let y_pred = X_test.model.value.softmax.argmax(axis = 1).indices
  echo accuracy_score(y_test, y_pred)
  echo "\n"
