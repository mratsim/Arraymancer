import ../src/arraymancer, random

# This is an early minimum viable example of handwritten digits recognition.
# It uses convolutional neural networks to achieve high accuracy.
#
# Data files (MNIST) can be downloaded here http://yann.lecun.com/exdb/mnist/
# and must be decompressed in "./build/" (or change the path "build/..." below)
#

# Make the results reproducible by initializing a random seed
randomize(42)

let
  ctx = newContext Tensor[float32] # Autograd/neural network graph
  n = 32                           # Batch size

let
  mnist = load_mnist(cache = true)
  # Training data is 60k 28x28 greyscale images from 0-255,
  # neural net prefers input rescaled to [0, 1] or [-1, 1]
  x_train = mnist.train_images.astype(float32) / 255'f32

  # Change shape from [N, H, W] to [N, C, H, W], with C = 1 (unsqueeze). Convolution expect 4d tensors
  # And store in the context to track operations applied and build a NN graph
  X_train = ctx.variable x_train.unsqueeze(1)

  # Labels are uint8, we must convert them to int
  y_train = mnist.train_labels.astype(int)

  # Idem for testing data (10000 images)
  x_test = mnist.test_images.astype(float32) / 255'f32
  X_test = ctx.variable x_test.unsqueeze(1)
  y_test = mnist.test_labels.astype(int)

# Configuration of the neural network
network ctx, DemoNet:
  layers:
    x:          Input([1, 28, 28])
    cv1:        Conv2D(x.out_shape, 20, 5, 5)
    mp1:        MaxPool2D(cv1.out_shape, (2,2), (0,0), (2,2))
    cv2:        Conv2D(mp1.out_shape, 50, 5, 5)
    mp2:        MaxPool2D(cv2.out_shape, (2,2), (0,0), (2,2))
    fl:         Flatten(mp2.out_shape)
    hidden:     Linear(fl.out_shape, 500)
    classifier: Linear(500, 10)
  forward x:
    x.cv1.relu.mp1.cv2.relu.mp2.fl.hidden.relu.classifier

let model = ctx.init(DemoNet)

# Stochastic Gradient Descent (API will change)
let optim = model.optimizerSGD(learning_rate = 0.01'f32)

# Learning loop
for epoch in 0 ..< 5:
  for batch_id in 0 ..< X_train.value.shape[0] div n: # some at the end may be missing, oh well ...
    # minibatch offset in the Tensor
    let offset = batch_id * n
    let x = X_train[offset ..< offset + n, _]
    let target = y_train[offset ..< offset + n]

    # Running through the network and computing loss
    let clf = model.forward(x)
    let loss = clf.sparse_softmax_cross_entropy(target)

    if batch_id mod 200 == 0:
      # Print status every 200 batches
      echo "Epoch is: " & $epoch
      echo "Batch id: " & $batch_id
      echo "Loss is:  " & $loss.value

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
      let y_pred = model.forward(X_test[i*1000 ..< (i+1)*1000, _]).value.softmax.argmax(axis = 1).squeeze
      score += y_pred.accuracy_score(y_test[i*1000 ..< (i+1)*1000])

      loss += model.forward(X_test[i*1000 ..< (i+1)*1000, _]).sparse_softmax_cross_entropy(y_test[i*1000 ..< (i+1)*1000]).value.data[0]
    score /= 10
    loss /= 10
    echo "Accuracy: " & $(score * 100) & "%"
    echo "Loss:     " & $loss
    echo "\n"


############# Output ############

# Epoch is: 0
# Batch id: 0
# Loss is:  3.133937835693359
# Epoch is: 0
# Batch id: 200
# Loss is:  0.3546932339668274
# Epoch is: 0
# Batch id: 400
# Loss is:  0.1979422867298126
# Epoch is: 0
# Batch id: 600
# Loss is:  0.1619873046875
# Epoch is: 0
# Batch id: 800
# Loss is:  0.1561944484710693
# Epoch is: 0
# Batch id: 1000
# Loss is:  0.2481455355882645
# Epoch is: 0
# Batch id: 1200
# Loss is:  0.1929974257946014
# Epoch is: 0
# Batch id: 1400
# Loss is:  0.09381495416164398
# Epoch is: 0
# Batch id: 1600
# Loss is:  0.08794669061899185
# Epoch is: 0
# Batch id: 1800
# Loss is:  0.2013712525367737

# Epoch #0 done. Testing accuracy
# Accuracy: 97.18000000000001%
# Loss:     0.09510207045823335


# Epoch is: 1
# Batch id: 0
# Loss is:  0.05660493671894073
# Epoch is: 1
# Batch id: 200
# Loss is:  0.05254033207893372
# Epoch is: 1
# Batch id: 400
# Loss is:  0.09177093207836151
# Epoch is: 1
# Batch id: 600
# Loss is:  0.0544213205575943
# Epoch is: 1
# Batch id: 800
# Loss is:  0.03129085898399353
# Epoch is: 1
# Batch id: 1000
# Loss is:  0.1740589588880539
# Epoch is: 1
# Batch id: 1200
# Loss is:  0.1218579858541489
# Epoch is: 1
# Batch id: 1400
# Loss is:  0.04907236993312836
# Epoch is: 1
# Batch id: 1600
# Loss is:  0.04116201400756836
# Epoch is: 1
# Batch id: 1800
# Loss is:  0.1408360302448273

# Epoch #1 done. Testing accuracy
# Accuracy: 98.00000000000001%
# Loss:     0.06425597975030542
