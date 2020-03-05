# Copyright 2017 the Arraymancer contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import ../../src/arraymancer, unittest, random

# Fix random seed for reproducibility
randomize(1234)

suite "[NN primitives] Loss functions":
  proc `~=`[T: SomeFloat](a, b: T): bool =
    let eps = 2e-5.T
    result = abs(a - b) <= eps

  test "Softmax cross-entropy & sparse softmax cross-entropy":

    block: # Simple test, no batch
      # https://www.pyimagesearch.com/2016/09/12/softmax-classifiers-explained/

      # Creating tensor of shape [batchsize, features]
      let predicted = [-3.44, 1.16, -0.81, 3.91].toTensor.reshape(1,4)
      let truth = [0'f64, 0, 0, 1].toTensor.reshape(1,4)

      let sce_loss = predicted.softmax_cross_entropy(truth)
      check: sce_loss ~= 0.0709

      let sparse_truth = [3].toTensor

      let sparse_sce_loss = predicted.sparse_softmax_cross_entropy(sparse_truth)
      check: sparse_sce_loss ~= 0.0709


      ## Test the gradient, create closures first:
      proc sce(pred: Tensor[float]): float =
        pred.softmax_cross_entropy(truth)

      proc sparse_sce(pred: Tensor[float]): float =
        pred.sparse_softmax_cross_entropy(sparse_truth)

      let expected_grad = sce_loss * numerical_gradient(predicted, sce)
      let expected_sparse_grad = sparse_sce_loss * numerical_gradient(predicted, sparse_sce)

      check: mean_relative_error(expected_grad, expected_sparse_grad) < 1e-6

      let grad = softmax_cross_entropy_backward(sce_loss, predicted, truth)
      check: grad.mean_relative_error(expected_grad) < 1e-6


      let sparse_grad = sparse_softmax_cross_entropy_backward(sparse_sce_loss, predicted, sparse_truth)
      check: sparse_grad.mean_relative_error(expected_sparse_grad) < 1e-6

    block: # with batch
      let batch_size = 256
      let nb_classes = 20

      # Create a sparse label tensor of shape: [batch_size]
      let sparse_labels = randomTensor(batch_size, nb_classes-1)

      # Create the corresponding dense label tensor of shape [batch_size, nb_classes]
      var labels = zeros[float64](batch_size, nb_classes)

      # Fill in the non-zeros values
      for sample_id, nonzero_idx in enumerate(sparse_labels):
        labels[sample_id, nonzero_idx] = 1

      # Create a random tensor with predictions:
      let pred = randomTensor(batch_size, nb_classes, -1.0..1.0)

      let sce_loss = pred.softmax_cross_entropy(labels)
      let sparse_sce_loss = pred.sparse_softmax_cross_entropy(sparse_labels)

      check: sce_loss ~= sparse_sce_loss

      ## Test the gradient, create closures first:
      proc sce(pred: Tensor[float]): float =
        pred.softmax_cross_entropy(labels)

      proc sparse_sce(pred: Tensor[float]): float =
        pred.sparse_softmax_cross_entropy(sparse_labels)

      let expected_grad = sce_loss * numerical_gradient(pred, sce)
      let expected_sparse_grad = sparse_sce_loss * numerical_gradient(pred, sparse_sce)

      check: mean_relative_error(expected_grad, expected_sparse_grad) < 1e-6

      let grad = softmax_cross_entropy_backward(sce_loss, pred, labels)
      check: grad.mean_relative_error(expected_grad) < 1e-6

      let sparse_grad = sparse_softmax_cross_entropy_backward(sparse_sce_loss, pred, sparse_labels)
      check: sparse_grad.mean_relative_error(expected_sparse_grad) < 1e-6
