# Copyright (c) 2017-2018 the Arraymancer contributors
# Licensed under the Apache License, version 2.0, ([LICENSE-APACHE](LICENSE) or http://www.apache.org/licenses/LICENSE-2.0)
# This file may not be copied, modified, or distributed except according to those terms.

import ../../src/arraymancer
import unittest, random

proc ex01() =
  let ctx = newContext Tensor[float32]

  let bsz = 32
  let x_train_bool = randomTensor([bsz * 10, 2], 1).astype(bool)
  let y_bool = x_train_bool[_,0] xor x_train_bool[_,1]

  let x_train = ctx.variable(x_train_bool.astype(float32))
  let y = y_bool.astype(float32)

  let layer_3neurons = ctx.variable(
                        randomTensor(3, 2, 2.0f) -. 1.0f,
                        true
                        )

  let classifier_layer = ctx.variable(
                    randomTensor(1, 3, 2.0f) -. 1.0f,
                    true
                    )

  let optim = newSGD[float32](
    layer_3neurons, classifier_layer, 0.01f # 0.01 is the learning rate
  )

  for epoch in 0..1:
    for batch_id in 0..<10:

      let offset = batch_id * 32
      let x = x_train[offset ..< offset + 32, _]
      let target = y[offset ..< offset + 32, _]

      let n1 = relu linear(x, layer_3neurons)
      let n2 = linear(n1, classifier_layer)
      let loss = n2.sigmoid_cross_entropy(target)

      # echo "Epoch is:" & $epoch
      # echo "Batch id:" & $batch_id
      # echo "Loss is:" & $loss.value

      loss.backprop()

      optim.update()

#########################################################################

proc ex02() =
  randomize(42)

  let
    ctx = newContext Tensor[float32]
    n = 32

  let
    mnist = load_mnist(cache = true)
    x_train = mnist.train_images.astype(float32) / 255'f32
    X_train = ctx.variable x_train.unsqueeze(1)
    y_train = mnist.train_labels.astype(int)
    x_test = mnist.test_images.astype(float32) / 255'f32
    X_test = ctx.variable x_test.unsqueeze(1)
    y_test = mnist.test_labels.astype(int)

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

  let optim = model.optimizerSGD(learning_rate = 0.01'f32)

  for epoch in 0 ..< 1:
    for batch_id in 0 ..< 1:
      let offset = batch_id * n
      let x = X_train[offset ..< offset + n, _]
      let target = y_train[offset ..< offset + n]

      let clf = model.forward(x)
      let loss = clf.sparse_softmax_cross_entropy(target)

      loss.backprop()
      optim.update()

    # Validation (checking the accuracy/generalization of our model on unseen data)
    ctx.no_grad_mode:
      var score = 0.0
      var loss = 0.0
      for i in 0 ..< 1:
        let y_pred = model.forward(X_test[i*1000 ..< (i+1)*1000, _]).value.softmax.argmax(axis = 1).squeeze
        score += y_pred.accuracy_score(y_test[i*1000 ..< (i+1)*1000])

        loss += model.forward(X_test[i*1000 ..< (i+1)*1000, _]).sparse_softmax_cross_entropy(y_test[i*1000 ..< (i+1)*1000]).value.unsafe_raw_buf[0]
      score /= 10
      loss /= 10

suite "End-to-End: mini-examples run":
  test "Example 1: XOR Perceptron":
    ex01()
  test "Example 2: MNIST via Convolutional Neural Net":
    ex02()
