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

import ../../src/arraymancer, unittest


suite "Loss functions":
  proc `~=`[T: SomeReal](a, b: T): bool =
    let eps = 2e-5.T
    result = abs(a - b) <= eps

  test "Softmax cross-entropy & sparse softmax cross-entropy":
    # https://www.pyimagesearch.com/2016/09/12/softmax-classifiers-explained/

    # Reminder, for now batch_size is the innermost index
    let predicted = [-3.44, 1.16, -0.81, 3.91].toTensor.reshape(4,1)
    let truth = [0'f64, 0, 0, 1].toTensor.reshape(4,1)

    let sce_loss = softmax_cross_entropy(predicted, truth)
    check: sce_loss ~= 0.0709

    let sparse_truth = [3].toTensor.reshape(1,1)

    let sparse_sce_loss = sparse_softmax_cross_entropy(predicted, sparse_truth)
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
    check: mean_relative_error(grad, expected_grad) < 1e-6


    let sparse_grad = sparse_softmax_cross_entropy_backward(sparse_sce_loss, predicted, sparse_truth)
    check: mean_relative_error(sparse_grad, expected_sparse_grad) < 1e-6