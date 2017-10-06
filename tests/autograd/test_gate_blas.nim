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

import ../../src/[arraymancer, arraymancer_ag]
import unittest, sequtils

# # Differentiating through matmul:
# # See http://cs231n.stanford.edu/vecDerivs.pdf
# # And: https://danieltakeshi.github.io/2017/01/21/understanding-higher-order-local-gradient-computation-for-backpropagation-in-deep-neural-networks/
# # And: http://cs231n.stanford.edu/handouts/linear-backprop.pdf

# # If base op is C = X * W
# ∂C/∂X = previous_gradient * W.transpose
# ∂C/∂W = X.transpose * previous_gradient

# # If base op is C = W * X (our case)
# ∂C/∂X = W.transpose * previous_gradient
# ∂C/∂W = previous_gradient * X.transpose

suite "Autograd of basic operations":
  test "Gradient of matrix multiplication":

    let W = toSeq(1..8).toTensor.reshape(2,4).astype(float32)
    let X = toSeq(11..22).toTensor.reshape(4,3).astype(float32)

    let ctx = newContext Tensor[float32]

    let w_ag = ctx.variable(W)
    let x_ag = ctx.variable(X)

    let C = w_ag * x_ag

    C.backprop

    let grad_C = ones[float32](2,3)
    check: w_ag.grad == grad_C * X.transpose
    check: x_ag.grad == W.transpose * grad_C
