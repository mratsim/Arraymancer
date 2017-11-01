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

  test "Softmax cross-entropy & sparse softmax cross-entrop":
    # https://www.pyimagesearch.com/2016/09/12/softmax-classifiers-explained/

    # Reminder, for now batch_size is the innermost index
    let predicted = [-3.44'f32, 1.16, -0.81, 3.91].toTensor.reshape(4,1)
    let truth = [0'f32, 0, 0, 1].toTensor.reshape(4,1)

    check: softmax_cross_entropy(predicted, truth) ~= 0.0709'f32

    let sparse_truth = [3].toTensor.reshape(1,1)

    check: sparse_softmax_cross_entropy(predicted, sparse_truth) ~= 0.0709'f32