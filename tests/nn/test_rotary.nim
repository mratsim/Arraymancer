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
import std / [unittest, math]

proc main() =
  suite "Rotary position embeddings":
    test "Rotary embeddings preserve relative positions":
      let ctx = newContext Tensor[float64]
      let rope = RotaryEmbedding[float64].init(head_dim = 8)
      let q = randomTensor[float64]([1, 1, 1, 8], 1.0)
      let k = randomTensor[float64]([1, 1, 1, 8], 1.0)

      ctx.no_grad_mode:
        # dot(q at i, k at j) only depends on i - j
        var dots: seq[float64] = @[]
        for (i, j) in [(0, 0), (7, 7), (5, 3), (11, 9)]:
          let qr = apply_rotary(ctx.variable(q), rope.forward(1, offset = i))
          let kr = apply_rotary(ctx.variable(k), rope.forward(1, offset = j))
          dots.add (qr *. kr).value.sum

        check: abs(dots[0] - dots[1]) < 1e-12 # relative position 0
        check: abs(dots[2] - dots[3]) < 1e-12 # relative position 2

    test "Partial rotary embeddings":
      let ctx = newContext Tensor[float64]
      let rope = RotaryEmbedding[float64].init(head_dim = 8, rotary_dim = 4)
      let x = randomTensor[float64]([1, 1, 1, 8], 1.0)

      ctx.no_grad_mode:
        let r = apply_rotary(ctx.variable(x), rope.forward(1, offset = 3)).value

        # channels beyond rotary_dim pass through
        check: r[_, _, _, 4 .. 7] == x[_, _, _, 4 .. 7]

        # rotated channels match the closed form
        for i in 0 .. 1:
          let angle = 3.0 * pow(10000.0, -2.0 * float64(i) / 4.0)
          let (c, s) = (cos(angle), sin(angle))
          check: abs(r[0, 0, 0, 2 * i] - (x[0, 0, 0, 2 * i] * c - x[0, 0, 0, 2 * i + 1] * s)) < 1e-12
          check: abs(r[0, 0, 0, 2 * i + 1] - (x[0, 0, 0, 2 * i] * s + x[0, 0, 0, 2 * i + 1] * c)) < 1e-12

    test "Rotary embeddings generalize to any leading rank":
      let ctx = newContext Tensor[float64]
      let rope = RotaryEmbedding[float64].init(head_dim = 8)
      let x = randomTensor[float64]([2, 3, 8], 1.0)

      ctx.no_grad_mode:
        let freqs = rope.forward(3, offset = 5)
        let a = apply_rotary(ctx.variable(x), freqs).value
        let b = apply_rotary(ctx.variable(x.reshape(2, 1, 3, 8)), freqs).value.reshape(2, 3, 8)
        check: max(abs(a - b)) < 1e-12

    test "Rotary embeddings backward with numerical gradient":
      let ctx = newContext Tensor[float64]
      let rope = RotaryEmbedding[float64].init(head_dim = 4)
      let freqs = rope.forward(3, offset = 5)
      let x = randomTensor[float64]([2, 2, 3, 4], 1.0)
      let grad_out = randomTensor[float64]([2, 2, 3, 4], 1.0)
      let vx = ctx.variable(x, requires_grad = true)
      let loss = (apply_rotary(vx, freqs) *. ctx.variable(grad_out)).sum()
      loss.backprop()

      proc loss_fn(inp: Tensor[float64]): float64 =
        let c = newContext Tensor[float64]
        c.no_grad_mode:
          result = (apply_rotary(c.variable(inp), freqs).value *. grad_out).sum

      let exp_grad = numerical_gradient(x, loss_fn)
      check: vx.grad.mean_relative_error(exp_grad) < 1e-6

  echo ""

main()
