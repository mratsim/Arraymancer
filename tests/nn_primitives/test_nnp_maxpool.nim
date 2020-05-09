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


import ../../src/arraymancer, ../testutils
import unittest

suite "[NN Primitives] Maxpool":
  let a =  [[1, 1, 2, 4],
            [5, 6, 7, 8],
            [3, 2, 1, 0],
            [1, 2, 3, 4]].toTensor.reshape(1,1,4,4)

  let (max_indices, pooled) = maxpool2d(a, (2,2), (0,0), (2,2))

  test "Maxpool2D forward":

    check: pooled == [6, 8, 3, 4].toTensor.reshape(1, 1, 2, 2)
    check: max_indices == [5, 7, 8, 15].toTensor

  test "Maxpool2D backward":

    # Create closure first
    proc mpool(t: Tensor[float]): float =
      maxpool2d(t, (2,2), (0,0), (2,2)).maxpooled.sum()

    let expected_grad = a.astype(float) *. numerical_gradient(a.astype(float), mpool)
    let grad = maxpool2d_backward(a.shape, max_indices, pooled).astype(float)

    check: grad.mean_relative_error(expected_grad) < 1e-6
