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

suite "LayerNorm":
  test "LayerNorm primitive without bias and autograd with numerical gradient":
    let ctx = newContext Tensor[float64]
    let x_t = randomTensor([2, 3, 4], 2.0) -. 1.0
    let w_t = randomTensor([4], 2.0)

    let x = ctx.variable(x_t, requires_grad = true)
    let w = ctx.variable(w_t, requires_grad = true)

    let y = x.layer_norm(w)
    check: y.value.shape == @[2, 3, 4]

    let loss = y.sum()
    loss.backprop()

    proc f_x(t: Tensor[float64]): float64 =
      layernorm(t, w_t).sum()
    let num_grad_x = x_t.numerical_gradient(f_x)
    check: x.grad.mean_relative_error(num_grad_x) < 1e-5

    proc f_w(t: Tensor[float64]): float64 =
      layernorm(x_t, t).sum()
    let num_grad_w = w_t.numerical_gradient(f_w)
    check: w.grad.mean_relative_error(num_grad_w) < 1e-5

  test "LayerNorm primitive with bias and autograd with numerical gradient":
    let ctx = newContext Tensor[float64]
    let x_t = randomTensor([2, 3, 4], 2.0) -. 1.0
    let w_t = randomTensor([4], 2.0)
    let b_t = randomTensor([4], 2.0)

    let x = ctx.variable(x_t, requires_grad = true)
    let w = ctx.variable(w_t, requires_grad = true)
    let b = ctx.variable(b_t, requires_grad = true)

    let y = x.layer_norm(w, b)
    check: y.value.shape == @[2, 3, 4]

    let loss = y.sum()
    loss.backprop()

    proc f_x(t: Tensor[float64]): float64 =
      layernorm(t, w_t, b_t).sum()
    check: x.grad.mean_relative_error(x_t.numerical_gradient(f_x)) < 1e-5

    proc f_w(t: Tensor[float64]): float64 =
      layernorm(x_t, t, b_t).sum()
    check: w.grad.mean_relative_error(w_t.numerical_gradient(f_w)) < 1e-5

    proc f_b(t: Tensor[float64]): float64 =
      layernorm(x_t, w_t, t).sum()
    check: b.grad.mean_relative_error(b_t.numerical_gradient(f_b)) < 1e-5

  test "LayerNorm layer (bias off by default)":
    let ctx = newContext Tensor[float32]
    let ln = ctx.init(LayerNorm[float32], normalized_shape = 8)

    check: ln.bias.isNil

    let x = ctx.variable(randomTensor([2, 5, 8], 1.0f), requires_grad = true)
    let y = ln.forward(x)
    check: y.value.shape == @[2, 5, 8]

    let loss = y.sum()
    loss.backprop()

    check: x.grad.shape == @[2, 5, 8]
    check: ln.weight.grad.shape == @[8]

  test "LayerNorm layer with bias":
    let ctx = newContext Tensor[float32]
    let ln = ctx.init(LayerNorm[float32], normalized_shape = 8, bias = true)

    check: not ln.bias.isNil

    let x = ctx.variable(randomTensor([2, 5, 8], 1.0f), requires_grad = true)
    let y = ln.forward(x)
    check: y.value.shape == @[2, 5, 8]

    let loss = y.sum()
    loss.backprop()

    check: x.grad.shape == @[2, 5, 8]
    check: ln.weight.grad.shape == @[8]
    check: ln.bias.grad.shape == @[8]
