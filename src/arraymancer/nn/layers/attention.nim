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
       ./rotary,
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

# key/value cache for autoregressive (incremental) inference

type KVCache*[T] = object
  k*: Tensor[T] # [batch, kv_heads, seq, head_dim]
  v*: Tensor[T]

proc isEmpty*[T](cache: KVCache[T]): bool =
  cache.k.size == 0

proc past_len*[T](cache: KVCache[T]): int =
  ## Number of keys/values already cached
  if cache.isEmpty: 0
  else: cache.k.shape[2]

proc repeat_kv*[T](
  x: Variable[Tensor[T]],
  repeats: int
): Variable[Tensor[T]] =
  ## Grouped query attention helper: expand `[batch, kv_heads, seq, head_dim]`
  ## by repeating each head `repeats` times, so that consecutive query heads
  ## share one key/value head.
  doAssert repeats >= 1, "repeats must be at least 1, got " & $repeats
  if repeats == 1: return x
  let g = x.value.shape[1]
  var expanded: seq[Variable[Tensor[T]]] = @[]
  for head in x.chunk(g, axis = 1):
    for _ in 0 ..< repeats:
      expanded.add head
  result = stack(expanded, axis = 1).squeeze(2)

# attention masks

proc promote_mask[T](mask: Tensor[T]): Tensor[T] =
  ## Left-pad an attention mask up to `[batch, heads, seq, key]`.
  if mask.size == 0: return mask
  doAssert mask.rank <= 4, "attention mask has at most 4 dimensions"
  result = mask
  while result.rank < 4:
    result = result.unsqueeze(0)

proc key_mask_additive[T: SomeFloat](
    key_mask: Tensor[bool],
    n_key: int,
    mask_val: T
): Tensor[T] =
  ## Convert a boolean `[batch, key]` key mask (`true` = masked out)
  ## into an additive `[batch, 1, 1, key]` attention mask.
  doAssert key_mask.rank == 2, "key_mask must have shape [batch, key]"
  doAssert key_mask.shape[1] == n_key, "key_mask does not match the number of keys"
  result = zeros[T]([key_mask.shape[0], 1, 1, n_key])
  for b in 0 ..< key_mask.shape[0]:
    for j in 0 ..< n_key:
      if key_mask[b, j]:
        result[b, 0, 0, j] = mask_val

# layers

proc validate_attention_dims(layer: string, embed_dim, num_heads, head_dim: int) =
  if embed_dim <= 0:
    raise newException(ValueError, layer & " embed_dim must be positive, got " & $embed_dim)
  if num_heads <= 0:
    raise newException(ValueError, layer & " num_heads must be positive, got " & $num_heads)
  if head_dim <= 0:
    raise newException(ValueError, layer & " head_dim must be positive, got " & $head_dim)

type MultiHeadAttention*[T] = object
  embed_dim*: int
  context_dim*: int # key/value feature dim, defaults to embed_dim
  num_heads*: int
  kv_heads*: int # key/value heads, defaults to num_heads (GQA/MQA)
  head_dim*: int
  dim_inner*: int # head_dim * num_heads
  q_proj*: Linear[T]
  k_proj*: Linear[T]
  v_proj*: Linear[T]
  out_proj*: Linear[T]

proc init*[T](
  ctx: Context[Tensor[T]],
  layerType: typedesc[MultiHeadAttention[T]],
  embed_dim, num_heads, head_dim: int,
  context_dim = 0,
  kv_heads = 0
): MultiHeadAttention[T] =
  validate_attention_dims("MultiHeadAttention", embed_dim, num_heads, head_dim)
  if context_dim < 0:
    raise newException(ValueError, "MultiHeadAttention context_dim must be non-negative, got " & $context_dim)
  let n_kv = if kv_heads == 0: num_heads else: kv_heads
  if n_kv < 1 or n_kv > num_heads:
    raise newException(ValueError,
      "MultiHeadAttention kv_heads must be in 1 .. " & $num_heads & ", got " & $kv_heads)
  if num_heads mod n_kv != 0:
    raise newException(ValueError,
      "MultiHeadAttention num_heads must be a multiple of kv_heads, got " & $num_heads & " and " & $n_kv)

  result.embed_dim = embed_dim
  result.context_dim = if context_dim == 0: embed_dim else: context_dim
  result.num_heads = num_heads
  result.kv_heads = n_kv
  result.head_dim = head_dim
  result.dim_inner = head_dim * num_heads

  result.q_proj = ctx.init(Linear[T], embed_dim, result.dim_inner, bias = false)
  result.k_proj = ctx.init(Linear[T], result.context_dim, head_dim * n_kv, bias = false)
  result.v_proj = ctx.init(Linear[T], result.context_dim, head_dim * n_kv, bias = false)
  result.out_proj = ctx.init(Linear[T], result.dim_inner, embed_dim, bias = false)

proc forward*[T](
  self: MultiHeadAttention[T],
  x: Variable[Tensor[T]],
  context: Variable[Tensor[T]] = nil,
  attn_mask: Tensor[T] = default(Tensor[T]),
  is_causal: bool = false,
  key_mask: Tensor[bool] = default(Tensor[bool]),
  rope: RotaryFreqs[T] = default(RotaryFreqs[T]),
  past: KVCache[T]
): tuple[output: Variable[Tensor[T]], present: KVCache[T]] =
  ## Incremental forward. Without `context`, queries, keys and values all come
  ## from `x` (self-attention) and new keys/values are appended to `past`.
  ## With a `context`, queries come from `x` while keys/values are projected
  ## from `context` (cross-attention), once, then reused from `past`.
  ##
  ## The cache is detached memory: with a non-empty `past` the whole
  ## key/value path is detached, so this path is meant for inference only.
  ## Backprop through a forward with an empty cache is fully supported.
  ##
  ## `attn_mask` is additive and broadcast from `[key]`, `[seq, key]`,
  ## `[heads, seq, key]` or `[batch, heads, seq, key]`. `key_mask` is a
  ## boolean `[batch, key]` mask where `true` masks the key out.
  ## `rope` holds the frequencies of the fresh positions: queries and new keys
  ## are rotated before they are cached, cached keys are used as-is. In
  ## cross-attention the context keys share the query frequencies, so the
  ## context must be exactly as long as `x`.
  ## Causal masking applies to self-attention only.
  ##
  ## Keys/values are projected with `kv_heads` heads and repeated to
  ## `num_heads` for attention (grouped query attention).

  let (b, n, h, d) = (x.value.shape[0], x.value.shape[1], self.num_heads, self.head_dim)
  let cross = not context.isNil
  let has_rope = not rope.isEmpty

  # queries
  var q = self.q_proj.forward(x).reshape(b, n, h, d).permute(0, 2, 1, 3)
  if has_rope:
    q = apply_rotary(q, rope)

  var k, v: Variable[Tensor[T]]
  var n_kv: int
  if cross:
    if is_causal:
      raise newException(ValueError, "MultiHeadAttention cannot be causal in cross-attention mode")
    # keys/values from the context, projected once
    if past.isEmpty:
      let (c_b, c_n) = (context.value.shape[0], context.value.shape[1])
      k = self.k_proj.forward(context).reshape(c_b, c_n, self.kv_heads, d).permute(0, 2, 1, 3)
      v = self.v_proj.forward(context).reshape(c_b, c_n, self.kv_heads, d).permute(0, 2, 1, 3)
      n_kv = c_n
      if has_rope:
        k = apply_rotary(k, rope)
    else:
      k = x.context.variable(past.k)
      v = x.context.variable(past.v)
      n_kv = past.past_len
  else:
    let past_n = past.past_len

    # keys/values from x
    k = self.k_proj.forward(x).reshape(b, n, self.kv_heads, d).permute(0, 2, 1, 3)
    v = self.v_proj.forward(x).reshape(b, n, self.kv_heads, d).permute(0, 2, 1, 3)
    if has_rope:
      k = apply_rotary(k, rope)
    n_kv = past_n + n

    # continue from the rotated cached prefix
    if past_n > 0:
      k = x.context.variable(concat(past.k, k.value, axis = 2))
      v = x.context.variable(concat(past.v, v.value, axis = 2))

  # attention mask, causal mask offset by the cached prefix
  var m = promote_mask(attn_mask)
  if is_causal:
    let causal = causal_mask[T](n, n_kv, past.past_len)
    m = if m.size == 0: causal else: m +. causal
  if key_mask.size > 0:
    let km = key_mask_additive[T](key_mask, n_kv, -1e9.T)
    m = if m.size == 0: km else: m +. km

  # grouped query attention: repeat each kv head for its query heads
  let repeats = self.num_heads div self.kv_heads
  let k_attn = if repeats == 1: k else: repeat_kv(k, repeats)
  let v_attn = if repeats == 1: v else: repeat_kv(v, repeats)

  # attention
  let output = scaled_dot_product_attention(q, k_attn, v_attn, mask = m)

  # merge heads
  let merged = output.permute(0, 2, 1, 3).reshape(b, n, self.dim_inner)

  # out
  result.output = self.out_proj.forward(merged)
  result.present = KVCache[T](k: k.value, v: v.value)

proc forward*[T](
  self: MultiHeadAttention[T],
  x: Variable[Tensor[T]],
  context: Variable[Tensor[T]] = nil,
  attn_mask: Tensor[T] = default(Tensor[T]),
  is_causal: bool = false,
  key_mask: Tensor[bool] = default(Tensor[bool]),
  rope: RotaryFreqs[T] = default(RotaryFreqs[T])
): Variable[Tensor[T]] =
  self.forward(x, context, attn_mask, is_causal, key_mask, rope, default(KVCache[T])).output
