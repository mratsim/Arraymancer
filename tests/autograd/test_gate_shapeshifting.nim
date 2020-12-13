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

import ../../src/arraymancer, ../testutils
import unittest, random, sequtils

proc main() =
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
        va.grad.mean_relative_error(target_grad_a) < 1e-07
        vb.grad.mean_relative_error(target_grad_b) < 1e-07
        vc.grad.mean_relative_error(target_grad_c) < 1e-07
        vd.grad.mean_relative_error(target_grad_d) < 1e-07

    test "Gradient of chunk operation":
      let
        height = rand(1..20)
        width = rand(1..20)

        x = randomTensor([4 * height, width], 1.0)

      proc network(x: Tensor[float64]): float64 =
        let s = x.chunk(4, axis = 0)
        check:
          s[0].shape == [height, width]
          s[1].shape == [height, width]
          s[2].shape == [height, width]
          s[3].shape == [height, width]
        result = sum(s[0] + s[1] - (s[2] + s[3]))

      let
        expected = x.clone().numerical_gradient(network)
        ctx = newContext Tensor[float64]
        vx = ctx.variable(x, requires_grad = true)

      let
        vs = vx.chunk(4, axis = 0)
        loss = sum(vs[0] + vs[1] - (vs[2] + vs[3]))

      loss.backprop()
      check: vx.grad.mean_relative_error(expected) < 1e-07

    test "Gradient of uneven chunks + slicing operations":

      # We split [10, width] tensors into 4 chunks along the first dim
      # We should have:
      #   - 2x [3, width] tensors
      #   - 2x [2, width] tensors
      # Then we slice each with t[0 ..< 2, _]

      let
        width = rand(1..20)

        x = randomTensor([10, width], 1.0)

      proc network(x: Tensor[float64]): float64 =
        let s = x.chunk(4, axis = 0)

        check:
          s[0].shape == [3, width]
          s[1].shape == [3, width]
          s[2].shape == [2, width]
          s[3].shape == [2, width]

        result = sum(
                    s[0][0 ..< 2, _] +
                    s[1][0 ..< 2, _] -
                    (
                      s[2][0 ..< 2, _] +
                      s[3][0 ..< 2, _]
                    )
                  )

      let
        expected = x.clone().numerical_gradient(network)
        ctx = newContext Tensor[float64]
        vx = ctx.variable(x, requires_grad = true)

      let
        vs = vx.chunk(4, axis = 0)
        loss = sum(
                  vs[0][0 ..< 2, _] +
                  vs[1][0 ..< 2, _] -
                  (
                    vs[2][0 ..< 2, _] +
                    vs[3][0 ..< 2, _]
                  )
                )

      loss.backprop()
      check: vx.grad.mean_relative_error(expected) < 1e-07

    test "Gradient of squeeze operation (+ chunking)":

      let
        M = rand(1..20)
        N = rand(1..20)

        x = randomTensor([1, M, N], 1.0)
        y = randomTensor([M, 1, N], 1.0)
        z = randomTensor([N, M, 3], 1.0)

      proc network_x(x: Tensor[float64]): float64 =
        result = 0
        for t in z.chunk(3, axis = 2):
          result += sum(
            (x.squeeze(0) + y.squeeze(1)) * t.squeeze(2)
          )
      proc network_y(y: Tensor[float64]): float64 =
        result = 0
        for t in z.chunk(3, axis = 2):
          result += sum(
            (x.squeeze(0) + y.squeeze(1)) * t.squeeze(2)
          )
      proc network_z(z: Tensor[float64]): float64 =
        result = 0
        for t in z.chunk(3, axis = 2):
          result += sum(
            (x.squeeze(0) + y.squeeze(1)) * t.squeeze(2)
          )

      let
        expected_x = x.clone().numerical_gradient(network_x)
        expected_y = y.clone().numerical_gradient(network_y)
        expected_z = z.clone().numerical_gradient(network_z)
        ctx = newContext Tensor[float64]
        vx = ctx.variable(x, requires_grad = true)
        vy = ctx.variable(y, requires_grad = true)
        vz = ctx.variable(z, requires_grad = true)

      # TODO: ease the following construct with variables
      let chunked = vz.chunk(3, axis = 2)
      var loss = sum(
        (vx.squeeze(0) + vy.squeeze(1)) * chunked[0].squeeze(2)
      )
      for i in 1..2:
        loss = loss + sum(
          (vx.squeeze(0) + vy.squeeze(1)) * chunked[i].squeeze(2)
        )

      loss.backprop()
      check:
        vx.grad.mean_relative_error(expected_x) < 1e-07
        vy.grad.mean_relative_error(expected_y) < 1e-07
        vz.grad.mean_relative_error(expected_z) < 1e-07

    test "Gradient of unsqueeze operation":

      let
        M = rand(1..20)
        N = rand(1..20)

        x = randomTensor([M, N], 1.0)
        y = randomTensor([M, 1, N], 1.0)

      proc network_x(x: Tensor[float64]): float64 =
        result = sum x.unsqueeze(1) + y
      proc network_y(y: Tensor[float64]): float64 =
        result = sum x.unsqueeze(1) + y

      let
        expected_x = x.clone().numerical_gradient(network_x)
        expected_y = y.clone().numerical_gradient(network_y)
        ctx = newContext Tensor[float64]
        vx = ctx.variable(x, requires_grad = true)
        vy = ctx.variable(y, requires_grad = true)

        loss = sum vx.unsqueeze(1) + vy

      loss.backprop()
      check:
        vx.grad.mean_relative_error(expected_x) < 1e-07
        vy.grad.mean_relative_error(expected_y) < 1e-07

main()
GC_fullCollect()
