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

      # additive bias == additive mask on the scores; learnable as a Variable
      let vb = ctx.variable(randomTensor[float64]([1, 1, S, S], 1.0), requires_grad = true)
      let biased = scaled_dot_product_attention(q, k, v, mask = mask, bias = vb)
      let masked = scaled_dot_product_attention(q, k, v, mask = mask +. vb.value)
      check: max(abs(biased.value - masked.value)) < 1e-12

      # tensor-level primitive takes a constant bias
      let raw = scaled_dot_product_attention(q.value, k.value, v.value, mask = mask, bias = vb.value)
      check: max(abs(raw - masked.value)) < 1e-12

      (attn_out.sum() + biased.sum()).backprop()
      check: q.grad.shape == @[B, H, S, D]
      check: k.grad.shape == @[B, H, S, D]
      check: v.grad.shape == @[B, H, S, D]
      check: vb.grad.shape == @[1, 1, S, S]

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

      # additive bias broadcasts like a mask and is learnable as a Variable
      let vb = ctx.variable(randomTensor[float32]([6, 6], 1.0f), requires_grad = true)
      let biased = mha.forward(x, is_causal = true, bias = vb)
      let masked = mha.forward(x, is_causal = true, mask = vb.value)
      check: max(abs(biased.value - masked.value)) < 1e-9

      biased.sum().backprop()
      check: vb.grad.shape == @[6, 6]

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

    test "repeat_kv groups consecutive query heads":
      let ctx = newContext Tensor[float64]
      # [1, 4, 1, 1] heads holding 1, 2, 3, 4
      var x = newTensor[float64]([1, 4, 1, 1])
      for i in 0 ..< 4: x[0, i, 0, 0] = float64(i + 1)

      let r = repeat_kv(ctx.variable(x), 2).value
      check: r.shape == @[1, 8, 1, 1]
      for i in 0 ..< 8:
        check: r[0, i, 0, 0] == float64(i div 2 + 1)

    test "MultiHeadAttention grouped query attention":
      let ctx = newContext Tensor[float32]
      let mha = ctx.init(
        MultiHeadAttention[float32], embed_dim = 16, num_heads = 4, head_dim = 4, kv_heads = 2
      )

      check: mha.num_heads == 4
      check: mha.kv_heads == 2
      check: mha.q_proj.weight.value.shape == @[16, 16]
      check: mha.k_proj.weight.value.shape == @[8, 16]
      check: mha.v_proj.weight.value.shape == @[8, 16]
      check: mha.out_proj.weight.value.shape == @[16, 16]

      let x = ctx.variable(randomTensor([2, 6, 16], 1.0f), requires_grad = true)
      let output = mha.forward(x, is_causal = true)
      check: output.value.shape == @[2, 6, 16]

      let loss = output.sum()
      loss.backprop()
      check: x.grad.shape == @[2, 6, 16]
      check: mha.k_proj.weight.grad.shape == @[8, 16]
      check: mha.v_proj.weight.grad.shape == @[8, 16]

      # a single shared kv head is multi-query attention
      let mqa = ctx.init(
        MultiHeadAttention[float32], embed_dim = 16, num_heads = 4, head_dim = 4, kv_heads = 1
      )
      check: mqa.kv_heads == 1
      check: mqa.k_proj.weight.value.shape == @[4, 16]
      check: mqa.forward(x).value.shape == @[2, 6, 16]

      # kv heads must divide the query heads
      expect ValueError:
        discard ctx.init(MultiHeadAttention[float32], embed_dim = 16, num_heads = 4, head_dim = 4, kv_heads = 3)
      expect ValueError:
        discard ctx.init(MultiHeadAttention[float32], embed_dim = 16, num_heads = 4, head_dim = 4, kv_heads = 5)

      # cross-attention keeps a compact cache
      ctx.no_grad_mode:
        let (o, c) = mha.forward(
          ctx.variable(randomTensor[float32]([1, 3, 16], 1.0f)),
          context = ctx.variable(randomTensor[float32]([1, 5, 16], 1.0f)),
          past = default(KVCache[float32])
        )
        check: o.value.shape == @[1, 3, 16]
        check: c.k.shape == @[1, 2, 5, 4]
        check: c.seen == 5 # context tokens

    test "MultiHeadAttention cross-attention mode":
      let ctx = newContext Tensor[float32]
      let mha = ctx.init(MultiHeadAttention[float32], embed_dim = 16, num_heads = 4, head_dim = 4)

      check: mha.head_dim == 4
      check: mha.dim_inner == 16
      check: mha.context_dim == 16
      check: mha.q_proj.bias.isNil
      check: mha.k_proj.bias.isNil
      check: mha.v_proj.bias.isNil
      check: mha.out_proj.bias.isNil

      let x = ctx.variable(randomTensor([2, 6, 16], 1.0f), requires_grad = true)
      let cond = ctx.variable(randomTensor([2, 9, 16], 1.0f), requires_grad = true)
      let output = mha.forward(x, context = cond)
      check: output.value.shape == @[2, 6, 16]

      let loss = output.sum()
      loss.backprop()

      check: x.grad.shape == @[2, 6, 16]
      check: cond.grad.shape == @[2, 9, 16]
      check: mha.q_proj.weight.grad.shape == @[16, 16]
      check: mha.k_proj.weight.grad.shape == @[16, 16]
      check: mha.v_proj.weight.grad.shape == @[16, 16]
      check: mha.out_proj.weight.grad.shape == @[16, 16]

      # the same layer still does self-attention
      check: mha.forward(x).value.shape == @[2, 6, 16]

      # separate context feature dim
      let mha2 = ctx.init(
        MultiHeadAttention[float32], embed_dim = 16, num_heads = 4, head_dim = 4, context_dim = 24
      )
      check: mha2.context_dim == 24
      check: mha2.k_proj.weight.value.shape == @[16, 24]
      let cond2 = ctx.variable(randomTensor([2, 9, 24], 1.0f))
      check: mha2.forward(x, context = cond2).value.shape == @[2, 6, 16]

      # mixed dims only work with a context
      expect ValueError:
        discard mha2.forward(x)

      # cross-attention cannot be causal
      expect ValueError:
        discard mha.forward(x, context = cond, is_causal = true)

    test "Causal mask with offset":
      let m = causal_mask[float64](2, 5, offset = 3)
      check: m.shape == @[1, 1, 2, 5]
      # query 0 is at absolute position 3: keys 0..3 are visible
      check: m[0, 0, 0, 3] == 0.0
      check: m[0, 0, 0, 4] == -1e9
      # query 1 is at absolute position 4: keys 0..4 are visible
      check: m[0, 0, 1, 4] == 0.0

    test "MultiHeadAttention cached decoding matches full forward":
      let ctx = newContext Tensor[float64]
      let mha = ctx.init(MultiHeadAttention[float64], embed_dim = 16, num_heads = 4, head_dim = 4)
      const n = 7
      let x = randomTensor[float64]([1, n, 16], 1.0)

      ctx.no_grad_mode:
        let full = mha.forward(ctx.variable(x), is_causal = true).value

        # token by token, no mask needed
        var cache = default(KVCache[float64])
        var outs: seq[Tensor[float64]] = @[]
        for i in 0 ..< n:
          let (o, c) = mha.forward(ctx.variable(x[_, i .. i, _]), is_causal = true, past = cache)
          cache = c
          outs.add o.value
        check: cache.seen == n
        check: max(abs(full - concat(outs, axis = 1))) < 1e-9

        # chunks of 3, 2 and 2, with an offset causal mask
        let (o1, c1) = mha.forward(ctx.variable(x[_, 0 .. 2, _]), is_causal = true, past = default(KVCache[float64]))
        let (o2, c2) = mha.forward(ctx.variable(x[_, 3 .. 4, _]), is_causal = true, past = c1)
        let (o3, c3) = mha.forward(ctx.variable(x[_, 5 .. 6, _]), is_causal = true, past = c2)
        check: c1.seen == 3
        check: c3.seen == n
        check: max(abs(full - concat(@[o1.value, o2.value, o3.value], axis = 1))) < 1e-9

    test "MultiHeadAttention fully masked keys attend nothing":
      let ctx = newContext Tensor[float64]
      let mha = ctx.init(MultiHeadAttention[float64], embed_dim = 16, num_heads = 4, head_dim = 4)
      let x = randomTensor[float64]([1, 3, 16], 1.0)

      var key_mask = newTensor[bool]([1, 3])
      for j in 0 ..< 3: key_mask[0, j] = true

      ctx.no_grad_mode:
        let masked_out = mha.forward(ctx.variable(x), key_mask = key_mask).value
        check: max(abs(masked_out)) < 1e-12

    test "MultiHeadAttention cross-attention mask and key_mask":
      let ctx = newContext Tensor[float64]
      let mha = ctx.init(MultiHeadAttention[float64], embed_dim = 16, num_heads = 4, head_dim = 4)
      let x = randomTensor[float64]([1, 3, 16], 1.0)
      let cond = randomTensor[float64]([1, 5, 16], 1.0)

      # keys 3 and 4 are padding
      var key_mask = newTensor[bool]([1, 5])
      for j in 0 ..< 5: key_mask[0, j] = j >= 3

      # equivalent additive padding masks at every accepted rank
      var m1 = zeros[float64]([5]) # [key]
      m1[3] = -1e9; m1[4] = -1e9
      var m2 = zeros[float64]([3, 5]) # [seq, key]
      for i in 0 ..< 3:
        m2[i, 3] = -1e9; m2[i, 4] = -1e9
      var m3 = zeros[float64]([4, 3, 5]) # [heads, seq, key]
      for h in 0 ..< 4:
        for i in 0 ..< 3:
          m3[h, i, 3] = -1e9; m3[h, i, 4] = -1e9
      var m4 = zeros[float64]([1, 4, 3, 5]) # [batch, heads, seq, key]
      for h in 0 ..< 4:
        for i in 0 ..< 3:
          m4[0, h, i, 3] = -1e9; m4[0, h, i, 4] = -1e9

      ctx.no_grad_mode:
        # masking padding is the same as dropping the padded keys
        let out_key = mha.forward(ctx.variable(x), context = ctx.variable(cond), key_mask = key_mask).value
        let out_cut = mha.forward(ctx.variable(x), context = ctx.variable(cond[_, 0 .. 2, _])).value
        check: max(abs(out_key - out_cut)) < 1e-9
        for m in [m1, m2, m3, m4]:
          let res = mha.forward(ctx.variable(x), context = ctx.variable(cond), mask = m).value
          check: max(abs(res - out_cut)) < 1e-9

        # full attention matrix: every query attends key 0 only,
        # so the output no longer depends on the queries
        var only0 = zeros[float64]([3, 5])
        for i in 0 ..< 3:
          for j in 1 ..< 5: only0[i, j] = -1e9
        let a = mha.forward(ctx.variable(x), context = ctx.variable(cond), mask = only0).value
        let b = mha.forward(ctx.variable(randomTensor[float64]([1, 3, 16], 1.0)), context = ctx.variable(cond), mask = only0).value
        check: max(abs(a - b)) < 1e-12

        # key padding with a cached context
        var cache = default(KVCache[float64])
        let (o1, c1) = mha.forward(ctx.variable(x[_, 0 .. 0, _]), context = ctx.variable(cond), key_mask = key_mask, past = cache)
        let (o2, c2) = mha.forward(ctx.variable(x[_, 1 .. 2, _]), context = ctx.variable(cond), key_mask = key_mask, past = c1)
        check: c1.seen == 5
        check: c2.seen == 5 # context tokens are not counted twice
        check: max(abs(out_key - concat(@[o1.value, o2.value], axis = 1))) < 1e-9

    test "MultiHeadAttention key_mask with cache":
      let ctx = newContext Tensor[float64]
      let mha = ctx.init(MultiHeadAttention[float64], embed_dim = 16, num_heads = 4, head_dim = 4)
      const n = 5
      let x = randomTensor[float64]([1, n, 16], 1.0)

      # keys 3 and 4 are padding
      var km = newTensor[bool]([1, n])
      for j in 0 ..< n: km[0, j] = j >= 3

      ctx.no_grad_mode:
        let full = mha.forward(ctx.variable(x), is_causal = true, key_mask = km).value

        var cache = default(KVCache[float64])
        var outs: seq[Tensor[float64]] = @[]
        for i in 0 ..< n:
          var kmi = newTensor[bool]([1, i + 1])
          for j in 0 .. i: kmi[0, j] = j >= 3
          let (o, c) = mha.forward(ctx.variable(x[_, i .. i, _]), is_causal = true, key_mask = kmi, past = cache)
          cache = c
          outs.add o.value

        check: max(abs(full - concat(outs, axis = 1))) < 1e-9

    test "MultiHeadAttention with rope backward with numerical gradient":
      let ctx = newContext Tensor[float64]
      let mha = ctx.init(MultiHeadAttention[float64], embed_dim = 8, num_heads = 2, head_dim = 4)
      let rope = RotaryEmbedding[float64].init(head_dim = 4, rotary_dim = 2)
      let x = randomTensor[float64]([1, 3, 8], 1.0)
      let grad_out = randomTensor[float64]([1, 3, 8], 1.0)

      let vx = ctx.variable(x, requires_grad = true)
      let loss = (mha.forward(vx, is_causal = true, rope = rope.forward(3)) *. ctx.variable(grad_out)).sum()
      loss.backprop()

      proc loss_fn(inp: Tensor[float64]): float64 =
        ctx.no_grad_mode:
          result = (mha.forward(ctx.variable(inp), is_causal = true, rope = rope.forward(3)).value *. grad_out).sum

      let exp_grad = numerical_gradient(x, loss_fn)
      check: vx.grad.mean_relative_error(exp_grad) < 1e-6

    test "MultiHeadAttention with rope cached decoding matches full forward":
      let ctx = newContext Tensor[float64]
      const n = 9
      let x = randomTensor[float64]([1, n, 16], 1.0)

      for rotary_dim in [0, 2, 4]:
        let mha = ctx.init(MultiHeadAttention[float64], embed_dim = 16, num_heads = 4, head_dim = 4)
        let rope = RotaryEmbedding[float64].init(head_dim = 4, rotary_dim = rotary_dim)

        ctx.no_grad_mode:
          let full = mha.forward(ctx.variable(x), is_causal = true, rope = rope.forward(n))

          var cache = default(KVCache[float64])
          var outs: seq[Tensor[float64]] = @[]
          for i in 0 ..< n:
            # only the fresh position is rotated, cached keys keep theirs
            let (output, present) = mha.forward(
              ctx.variable(x[_, i .. i, _]), is_causal = true,
              rope = rope.forward(1, offset = i), past = cache
            )
            cache = present
            outs.add output.value

          check: cache.seen == n
          let seq_out = concat(outs, axis = 1)
          check: max(abs(full.value - seq_out)) < 1e-9

          # chunks of 3, 2 and 4, with an offset causal mask
          let (a1, c1) = mha.forward(ctx.variable(x[_, 0 .. 2, _]), is_causal = true, rope = rope.forward(3), past = default(KVCache[float64]))
          let (a2, c2) = mha.forward(ctx.variable(x[_, 3 .. 4, _]), is_causal = true, rope = rope.forward(2, offset = 3), past = c1)
          let (a3, c3) = mha.forward(ctx.variable(x[_, 5 .. 8, _]), is_causal = true, rope = rope.forward(4, offset = 5), past = c2)
          check: c3.seen == n
          check: max(abs(full.value - concat(@[a1.value, a2.value, a3.value], axis = 1))) < 1e-9

    test "MultiHeadAttention caches rotated keys":
      let ctx = newContext Tensor[float64]
      let mha = ctx.init(MultiHeadAttention[float64], embed_dim = 8, num_heads = 2, head_dim = 4)
      let rope = RotaryEmbedding[float64].init(head_dim = 4)
      let x = randomTensor[float64]([1, 2, 8], 1.0)
      let token = ctx.variable(x[_, 1 .. 1, _])

      ctx.no_grad_mode:
        let (_, cache) = mha.forward(token, rope = rope.forward(1, offset = 1), past = default(KVCache[float64]))

        # cached keys are the projections rotated at their absolute position
        let projected = mha.k_proj.forward(token).value.reshape(1, 1, mha.kv_heads, mha.head_dim).permute(0, 2, 1, 3)
        check: max(abs(cache.k - apply_rotary(ctx.variable(projected), rope.forward(1, offset = 1)).value)) < 1e-12

    test "MultiHeadAttention GQA cached decoding matches full forward":
      let ctx = newContext Tensor[float64]
      let rope = RotaryEmbedding[float64].init(head_dim = 4, rotary_dim = 2)
      const n = 9
      let x = randomTensor[float64]([1, n, 16], 1.0)

      for kv_heads in [1, 2]:
        let mha = ctx.init(MultiHeadAttention[float64], embed_dim = 16, num_heads = 4, head_dim = 4, kv_heads = kv_heads)

        ctx.no_grad_mode:
          let full = mha.forward(ctx.variable(x), is_causal = true, rope = rope.forward(n)).value

          var cache = default(KVCache[float64])
          var outs: seq[Tensor[float64]] = @[]
          for i in 0 ..< n:
            let (output, present) = mha.forward(
              ctx.variable(x[_, i .. i, _]), is_causal = true,
              rope = rope.forward(1, offset = i), past = cache
            )
            cache = present
            outs.add output.value

          check: cache.seen == n
          check: cache.k.shape == @[1, kv_heads, n, 4] # compact kv cache
          check: max(abs(full - concat(outs, axis = 1))) < 1e-9

    test "MultiHeadAttention GQA backward with numerical gradient":
      let ctx = newContext Tensor[float64]
      let rope = RotaryEmbedding[float64].init(head_dim = 2)
      let x = randomTensor[float64]([1, 3, 8], 1.0)
      let grad_out = randomTensor[float64]([1, 3, 8], 1.0)

      for kv_heads in [1, 2]:
        let mha = ctx.init(MultiHeadAttention[float64], embed_dim = 8, num_heads = 4, head_dim = 2, kv_heads = kv_heads)

        let vx = ctx.variable(x, requires_grad = true)
        let loss = (mha.forward(vx, is_causal = true, rope = rope.forward(3)) *. ctx.variable(grad_out)).sum()
        loss.backprop()

        proc loss_fn(inp: Tensor[float64]): float64 =
          ctx.no_grad_mode:
            result = (mha.forward(ctx.variable(inp), is_causal = true, rope = rope.forward(3)).value *. grad_out).sum

        let exp_grad = numerical_gradient(x, loss_fn)
        check: vx.grad.mean_relative_error(exp_grad) < 1e-6

main()
