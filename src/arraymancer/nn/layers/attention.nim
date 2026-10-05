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

import ../../tensor,
       ../../autograd,
       ../../nn_primitives,
       ../activation/softmax,
       ./linear,
       math

# attention

proc scaled_dot_product_attention*[TT](
    query, key, value: Variable[TT],
    scale: float64 = 0.0,
    mask: TT = default(TT)
): Variable[TT] =
  # scale
  let d = query.value.shape[^1]
  let s = if scale == 0.0: pow(d.float64, -0.5) else: scale

  # sim
  var sim = (query * key.transpose2d) * s

  # mask
  if mask.size > 0:
    sim = sim +. mask

  # attention
  let attn = softmax(sim, axis = -1)

  # out
  result = attn * value

type MultiHeadAttention*[T] = object
  embed_dim*: int
  num_heads*: int
  head_dim*: int
  dim_inner*: int
  q_proj*: Linear[T]
  k_proj*: Linear[T]
  v_proj*: Linear[T]
  out_proj*: Linear[T]

proc init*[T](
  ctx: Context[Tensor[T]],
  layerType: typedesc[MultiHeadAttention[T]],
  embed_dim, num_heads, head_dim: int
): MultiHeadAttention[T] =
  result.embed_dim = embed_dim
  result.num_heads = num_heads
  result.head_dim = head_dim
  result.dim_inner = head_dim * num_heads

  result.q_proj = ctx.init(Linear[T], embed_dim, result.dim_inner, bias = false)
  result.k_proj = ctx.init(Linear[T], embed_dim, result.dim_inner, bias = false)
  result.v_proj = ctx.init(Linear[T], embed_dim, result.dim_inner, bias = false)
  result.out_proj = ctx.init(Linear[T], result.dim_inner, embed_dim, bias = false)

proc forward*[T](
  self: MultiHeadAttention[T],
  x: Variable[Tensor[T]],
  mask: Tensor[T] = default(Tensor[T]),
  is_causal: bool = false
): Variable[Tensor[T]] =
  let (b, n, h, d) = (x.value.shape[0], x.value.shape[1], self.num_heads, self.head_dim)

  # queries, keys, values
  let q = self.q_proj.forward(x).reshape(b, n, h, d).permute(0, 2, 1, 3)
  let k = self.k_proj.forward(x).reshape(b, n, h, d).permute(0, 2, 1, 3)
  let v = self.v_proj.forward(x).reshape(b, n, h, d).permute(0, 2, 1, 3)

  # causal mask
  let m =
    if is_causal:
      let causal = causal_mask[T](n)
      if mask.size == 0: causal
      else: mask +. causal
    else: mask

  # attention
  let output = scaled_dot_product_attention(q, k, v, mask = m)

  # merge heads
  let merged = output.permute(0, 2, 1, 3).reshape(b, n, self.dim_inner)

  # out
  result = self.out_proj.forward(merged)
