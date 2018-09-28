# Copyright 2017-2018 Mamy André-Ratsimbazafy & the Arraymancer contributors
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
import unittest, random, sequtils

suite "Autograd of shapeshifting operations":
  test "Gradient of stack operation":
    let
      height = rand(1..20)
      width = rand(1..20)

    let
      a = randomTensor([height, width], 1.0)
      b = randomTensor([height, width], 1.0)
      c = randomTensor([height, width], 1.0)
      d = randomTensor([height, width], 1.0)

    proc stack_a(a: Tensor[float64]): float64 = stack(a, a + b, c - d, axis = 0).sum()
    proc stack_b(b: Tensor[float64]): float64 = stack(a, a + b, c - d, axis = 0).sum()
    proc stack_c(c: Tensor[float64]): float64 = stack(a, a + b, c - d, axis = 0).sum()
    proc stack_d(d: Tensor[float64]): float64 = stack(a, a + b, c - d, axis = 0).sum()

    let # Compute the numerical gradients
      target_grad_a = a.numerical_gradient(stack_a)
      target_grad_b = b.numerical_gradient(stack_b)
      target_grad_c = c.numerical_gradient(stack_c)
      target_grad_d = d.numerical_gradient(stack_d)

    let
      ctx = newContext Tensor[float64]
      va = ctx.variable(a, requires_grad = true)
      vb = ctx.variable(b, requires_grad = true)
      vc = ctx.variable(c, requires_grad = true)
      vd = ctx.variable(d, requires_grad = true)

    let loss = stack(va, va + vb, vc - vd, axis = 0).sum()
    loss.backprop()

    check:
      mean_relative_error(va.grad, target_grad_a) < 1e-07
      mean_relative_error(vb.grad, target_grad_b) < 1e-07
      mean_relative_error(vc.grad, target_grad_c) < 1e-07
      mean_relative_error(vd.grad, target_grad_d) < 1e-07
