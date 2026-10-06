# Copyright 2026 the Arraymancer contributors
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

import ../../src/arraymancer
import std / unittest

suite "RMSNorm":
  test "RMSNorm primitive and autograd with numerical gradient":
    let ctx = newContext Tensor[float64]
    let x_t = randomTensor([2, 3, 4], 2.0) -. 1.0
    let w_t = randomTensor([4], 2.0)

    let x = ctx.variable(x_t, requires_grad = true)
    let w = ctx.variable(w_t, requires_grad = true)

    let y = x.rms_norm(w)
    check: y.value.shape == @[2, 3, 4]

    let loss = y.sum()
    loss.backprop()

    proc f_x(t: Tensor[float64]): float64 =
      rmsnorm(t, w_t).sum()
    let num_grad_x = x_t.numerical_gradient(f_x)
    check: x.grad.mean_relative_error(num_grad_x) < 1e-5

    proc f_w(t: Tensor[float64]): float64 =
      rmsnorm(x_t, t).sum()
    let num_grad_w = w_t.numerical_gradient(f_w)
    check: w.grad.mean_relative_error(num_grad_w) < 1e-5

  test "RMSNorm layer":
    let ctx = newContext Tensor[float32]
    let rms = ctx.init(RMSNorm[float32], normalized_shape = 8)

    let x = ctx.variable(randomTensor([2, 5, 8], 1.0f), requires_grad = true)
    let y = rms.forward(x)
    check: y.value.shape == @[2, 5, 8]

    let loss = y.sum()
    loss.backprop()

    check: x.grad.shape == @[2, 5, 8]
    check: rms.weight.grad.shape == @[8]
