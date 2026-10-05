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
import std / [unittest, sequtils, math]

proc main() =
  suite "Attention, Batched GEMM, and Transformer Primitives":
    test "transpose2d on Tensor and Variable":
      let t3 = randomTensor([2, 3, 4], 1.0f)
      let t3_t = t3.transpose2d
      check: t3_t.shape == @[2, 4, 3]
      check: t3_t.transpose2d == t3

      let ctx = newContext Tensor[float64]
      let v = ctx.variable(randomTensor([2, 3, 4], 1.0), requires_grad = true)
      let v_t = v.transpose2d
      check: v_t.value.shape == @[2, 4, 3]

      let loss = v_t.sum()
      loss.backprop()
      check: v.grad == ones[float64]([2, 3, 4])

    test "Batched Matrix Multiplication (BMM) forward":
      # 3D x 3D
      let a3 = randomTensor([2, 3, 4], 1.0f)
      let b3 = randomTensor([2, 4, 5], 1.0f)
      let c3 = a3 * b3
      check: c3.shape == @[2, 3, 5]
      check: c3[0, _, _].squeeze(0) == a3[0, _, _].squeeze(0) * b3[0, _, _].squeeze(0)
      check: c3[1, _, _].squeeze(0) == a3[1, _, _].squeeze(0) * b3[1, _, _].squeeze(0)

      # 4D x 4D (Multi-Head Attention shape: [Batch, Heads, Seq, HeadDim])
      let a4 = randomTensor([2, 4, 8, 16], 1.0f)
      let b4 = randomTensor([2, 4, 16, 8], 1.0f)
      let c4 = a4 * b4
      check: c4.shape == @[2, 4, 8, 8]

      # 3D x 2D (Token sequence times weight matrix)
      let b2 = randomTensor([4, 6], 1.0f)
      let c32 = a3 * b2
      check: c32.shape == @[2, 3, 6]
      check: c32[0, _, _].squeeze(0) == a3[0, _, _].squeeze(0) * b2

    test "BMM backward with numerical gradient":
      let a = randomTensor([2, 3, 4], 1.0)
      let b = randomTensor([2, 4, 5], 1.0)
      let grad_c = randomTensor([2, 3, 5], 1.0)

      let ctx = newContext Tensor[float64]
      let va = ctx.variable(a, requires_grad = true)
      let vb = ctx.variable(b, requires_grad = true)
      let vc = va * vb
      let loss = (vc *. ctx.variable(grad_c)).sum()
      loss.backprop()

      proc loss_a(inp: Tensor[float64]): float64 = (grad_c *. (inp * b)).sum
      proc loss_b(inp: Tensor[float64]): float64 = (grad_c *. (a * inp)).sum

      let exp_ga = numerical_gradient(a, loss_a)
      let exp_gb = numerical_gradient(b, loss_b)

      check: va.grad.mean_relative_error(exp_ga) < 1e-6
      check: vb.grad.mean_relative_error(exp_gb) < 1e-6

    test "Permute in Autograd":
      let ctx = newContext Tensor[float64]
      let x = randomTensor([2, 3, 4, 5], 1.0)
      let vx = ctx.variable(x, requires_grad = true)
      let permuted = vx.permute(0, 2, 1, 3)
      check: permuted.value.shape == @[2, 4, 3, 5]

      let grad_out = randomTensor([2, 4, 3, 5], 1.0)
      let loss = (permuted *. ctx.variable(grad_out)).sum()
      loss.backprop()

      check: vx.grad == grad_out.permute(0, 2, 1, 3)

    test "Softmax and Softmax Backward with numerical gradient":
      # 2D along last axis
      let x2 = randomTensor([3, 5], 1.0)
      let grad_out2 = randomTensor([3, 5], 1.0)
      let ctx2 = newContext Tensor[float64]
      let vx2 = ctx2.variable(x2, requires_grad = true)
      let vy2 = vx2.softmax(axis = -1)
      let loss2 = (vy2 *. ctx2.variable(grad_out2)).sum()
      loss2.backprop()

      proc loss_fn2(inp: Tensor[float64]): float64 = (grad_out2 *. inp.softmax(axis = -1)).sum
      let exp_grad2 = numerical_gradient(x2, loss_fn2)
      check: vx2.grad.mean_relative_error(exp_grad2) < 1e-6

      # 4D along last axis (like attention scores)
      let x4 = randomTensor([2, 3, 4, 5], 1.0)
      let grad_out4 = randomTensor([2, 3, 4, 5], 1.0)
      let ctx4 = newContext Tensor[float64]
      let vx4 = ctx4.variable(x4, requires_grad = true)
      let vy4 = vx4.softmax(axis = -1)
      let loss4 = (vy4 *. ctx4.variable(grad_out4)).sum()
      loss4.backprop()

      proc loss_fn4(inp: Tensor[float64]): float64 = (grad_out4 *. inp.softmax(axis = -1)).sum
      let exp_grad4 = numerical_gradient(x4, loss_fn4)
      check: vx4.grad.mean_relative_error(exp_grad4) < 1e-6

    test "Scale Gate (Variable * scalar)":
      let ctx = newContext Tensor[float64]
      let x = randomTensor([2, 3], 1.0)
      let vx = ctx.variable(x, requires_grad = true)
      let vy = vx * 2.5
      let loss = vy.sum()
      loss.backprop()
      check: vx.grad == ones[float64]([2, 3]) *. 2.5

    test "Broadcasted Addition & Mask Addition":
      let ctx = newContext Tensor[float64]
      let scores = ctx.variable(zeros[float64]([2, 2, 4, 4]), requires_grad = true)
      let mask = causal_mask[float64](4)
      let masked_scores = scores +. mask
      check: masked_scores.value.shape == @[2, 2, 4, 4]
      check: masked_scores.value[0, 0, 0, 1] == -1e9
      check: masked_scores.value[0, 0, 1, 0] == 0.0

      let loss = masked_scores.sum()
      loss.backprop()
      check: scores.grad == ones[float64]([2, 2, 4, 4])

    test "Linear layer on rank-3 tensor":
      let ctx = newContext Tensor[float64]
      let x = randomTensor([2, 5, 8], 1.0)
      let vx = ctx.variable(x, requires_grad = true)
      let lin = ctx.init(Linear[float64], 8, 16)
      let output = lin.forward(vx)
      check: output.value.shape == @[2, 5, 16]

      let loss = output.sum()
      loss.backprop()
      check: vx.grad.shape == @[2, 5, 8]
      check: lin.weight.grad.shape == @[16, 8]
      check: lin.bias.grad.shape == @[1, 16]

    test "Scaled Dot-Product Attention (primitive)":
      let ctx = newContext Tensor[float64]
      let B = 2
      let H = 3
      let S = 4
      let D = 8

      let q = ctx.variable(randomTensor([B, H, S, D], 1.0), requires_grad = true)
      let k = ctx.variable(randomTensor([B, H, S, D], 1.0), requires_grad = true)
      let v = ctx.variable(randomTensor([B, H, S, D], 1.0), requires_grad = true)

      let mask = causal_mask[float64](S)
      let attn_out = scaled_dot_product_attention(q, k, v, mask = mask)
      check: attn_out.value.shape == @[B, H, S, D]

      let loss = attn_out.sum()
      loss.backprop()

      check: q.grad.shape == @[B, H, S, D]
      check: k.grad.shape == @[B, H, S, D]
      check: v.grad.shape == @[B, H, S, D]

    test "MultiHeadAttention Layer with Causal Mask":
      let ctx = newContext Tensor[float32]
      let mha = ctx.init(MultiHeadAttention[float32], embed_dim = 16, num_heads = 4, head_dim = 4)

      check: mha.head_dim == 4
      check: mha.dim_inner == 16
      check: mha.q_proj.bias.isNil
      check: mha.k_proj.bias.isNil
      check: mha.v_proj.bias.isNil
      check: mha.out_proj.bias.isNil

      let x = ctx.variable(randomTensor([2, 6, 16], 1.0f), requires_grad = true)
      let output = mha.forward(x, is_causal = true)
      check: output.value.shape == @[2, 6, 16]

      let loss = output.sum()
      loss.backprop()

      check: x.grad.shape == @[2, 6, 16]
      check: mha.q_proj.weight.grad.shape == @[16, 16]
      check: mha.k_proj.weight.grad.shape == @[16, 16]
      check: mha.v_proj.weight.grad.shape == @[16, 16]

    test "MultiHeadAttention with explicit head_dim != embed_dim / heads":
      let ctx = newContext Tensor[float32]
      let mha = ctx.init(MultiHeadAttention[float32], embed_dim = 16, num_heads = 4, head_dim = 8)

      check: mha.head_dim == 8
      check: mha.dim_inner == 32

      let x = ctx.variable(randomTensor([2, 6, 16], 1.0f), requires_grad = true)
      let output = mha.forward(x, is_causal = true)
      check: output.value.shape == @[2, 6, 16]

      let loss = output.sum()
      loss.backprop()

      check: x.grad.shape == @[2, 6, 16]
      check: mha.q_proj.weight.grad.shape == @[32, 16]
      check: mha.k_proj.weight.grad.shape == @[32, 16]
      check: mha.v_proj.weight.grad.shape == @[32, 16]
      check: mha.out_proj.weight.grad.shape == @[16, 32]

main()
