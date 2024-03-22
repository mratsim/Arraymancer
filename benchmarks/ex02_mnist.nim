import ../src/arraymancer
import std / random

# Make the results reproducible by initializing a random seed
randomize(42)

let
  ctx = newContext Tensor[float32] # Autograd/neural network graph
  n = 32                           # Batch size

let
  mnist = load_mnist(cache = true)
  x_train = mnist.train_images.asType(float32) / 255'f32
  X_train = ctx.variable x_train.unsqueeze(1)
  y_train = mnist.train_labels.asType(int)

  x_test = mnist.test_images.asType(float32) / 255'f32
  X_test = ctx.variable x_test.unsqueeze(1)
  y_test = mnist.test_labels.asType(int)

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
let optim = model.optimizer(SGD, learning_rate = 0.01'f32)

# Learning loop
for epoch in 0 ..< 2:
  for batch_id in 0 ..< X_train.value.shape[0] div n: # some at the end may be missing
    let offset = batch_id * n
    let x = X_train[offset ..< offset + n, _]
    let target = y_train[offset ..< offset + n]

    let clf = model.forward(x)
    let loss = clf.sparse_softmax_cross_entropy(target)

    loss.backprop()
    optim.update()

  ctx.no_grad_mode:
    var score = 0.0
    var loss = 0.0
    for i in 0 ..< 10:
      let y_pred = model.forward(X_test[i*1000 ..< (i+1)*1000, _]).value.softmax.argmax(axis = 1).squeeze
      score += y_pred.accuracy_score(y_test[i*1000 ..< (i+1)*1000])

      loss += model.forward(X_test[i*1000 ..< (i+1)*1000, _]).sparse_softmax_cross_entropy(y_test[i*1000 ..< (i+1)*1000]).value.unsafe_raw_offset[0]
    score /= 10
    loss /= 10
