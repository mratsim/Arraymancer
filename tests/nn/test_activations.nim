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
  suite "GELU activation":
    test "GELU forward matches the closed form":
      let ctx = newContext Tensor[float64]
      let x = randomTensor[float64]([3, 4], 4.0) -. 2.0

      ctx.no_grad_mode:
        let y = gelu(ctx.variable(x)).value
        check: y.shape == @[3, 4]
        for i in 0 ..< 3:
          for j in 0 ..< 4:
            let e = 0.5 * x[i, j] * (1 + erf(x[i, j] / sqrt(2.0)))
            check: abs(y[i, j] - e) < 1e-12

    test "GELU backward with numerical gradient":
      let ctx = newContext Tensor[float64]
      let x = randomTensor[float64]([2, 5], 2.0) -. 1.0
      let grad_out = randomTensor[float64]([2, 5], 1.0)
      let vx = ctx.variable(x, requires_grad = true)
      let loss = (gelu(vx) *. ctx.variable(grad_out)).sum()
      loss.backprop()

      proc loss_fn(inp: Tensor[float64]): float64 =
        let c = newContext Tensor[float64]
        c.no_grad_mode:
          result = (gelu(c.variable(inp)).value *. grad_out).sum

      check: vx.grad.mean_relative_error(numerical_gradient(x, loss_fn)) < 1e-6

  echo ""

main()
