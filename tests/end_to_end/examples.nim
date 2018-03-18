# Copyright (c) 2017-2018 the Arraymancer contributors
# Licensed under the Apache License, version 2.0, ([LICENSE-APACHE](LICENSE) or http://www.apache.org/licenses/LICENSE-2.0)
# This file may not be copied, modified, or distributed except according to those terms.

import ../../src/arraymancer
import unittest, random

suite "End-to-End: Examples compile and run":
  test "Example 1: XOR Perceptron":
    let ctx = newContext Tensor[float32]

    let bsz = 32
    let x_train_bool = randomTensor([bsz * 10, 2], 1).astype(bool)
    let y_bool = x_train_bool[_,0] xor x_train_bool[_,1]

    let x_train = ctx.variable(x_train_bool.astype(float32))
    let y = y_bool.astype(float32)

    let layer_3neurons = ctx.variable(
                          randomTensor(3, 2, 2.0f) .- 1.0f,
                          requires_grad = true
                          )

    let classifier_layer = ctx.variable(
                      randomTensor(1, 3, 2.0f) .- 1.0f,
                      requires_grad = true
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
        let loss = sigmoid_cross_entropy(n2, target)

        # echo "Epoch is:" & $epoch
        # echo "Batch id:" & $batch_id
        # echo "Loss is:" & $loss.value.data[0]

        loss.backprop()

        optim.update()

