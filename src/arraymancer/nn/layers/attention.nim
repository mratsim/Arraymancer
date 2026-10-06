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
  ## `softmax(query keyᵀ * scale + mask) value`
  let d = query.value.shape[^1]
  let s = if scale == 0.0: pow(d.float64, -0.5) else: scale

  var sim = (query * key.transpose2d) * s
  if mask.size > 0:
    sim = sim +. mask

  softmax(sim, axis = -1) * value

# key/value cache for incremental inference

type KVCache*[T] = object
  seen*: int     # tokens seen, grows past the cached window when sliding
  k*: Tensor[T]  # [batch, kv_heads, cached_seq, head_dim]
  v*: Tensor[T]

proc isEmpty*[T](cache: KVCache[T]): bool =
  cache.k.size == 0

proc repeat_kv*[T](
  x: Variable[Tensor[T]],
  repeats: int
): Variable[Tensor[T]] =
  ## GQA: repeat each kv head for its `repeats` consecutive query heads
  doAssert repeats >= 1, "repeats must be at least 1, got " & $repeats
  if repeats == 1: return x
  var heads: seq[Variable[Tensor[T]]] = @[]
  for head in x.chunk(x.value.shape[1], axis = 1):
    for _ in 0 ..< repeats:
      heads.add head
  stack(heads, axis = 1).squeeze(2)

# attention masks

proc promote_mask[T](mask: Tensor[T]): Tensor[T] =
  ## Left-pad an attention mask up to `[batch, heads, seq, key]`.
  if mask.size == 0: return mask
  doAssert mask.rank <= 4, "attention mask has at most 4 dimensions"
  result = mask
  while result.rank < 4:
    result = result.unsqueeze(0)

proc key_mask_additive[T: SomeFloat](key_mask: Tensor[bool], mask_val: T): Tensor[T] =
  ## Boolean `[batch, key]` key mask (`true` = masked out)
  ## -> additive `[batch, 1, 1, key]` attention mask
  doAssert key_mask.rank == 2, "key_mask must have shape [batch, key]"
  result = map_inline(key_mask):
    if x: mask_val else: 0.T
  result = result.unsqueeze(1).unsqueeze(2)

proc key_visibility[T: SomeFloat](key_mask: Tensor[bool]): Tensor[T] =
  ## [batch, 1, 1, 1]: 0 when every key of the row is masked
  result = ones[T]([key_mask.shape[0], 1, 1, 1])
  for b in 0 ..< key_mask.shape[0]:
    var any_visible = false
    for j in 0 ..< key_mask.shape[1]:
      if not key_mask[b, j]: any_visible = true
    if not any_visible: result[b, 0, 0, 0] = 0.T

proc add_mask[T](base, extra: Tensor[T]): Tensor[T] =
  ## Combine additive masks, an empty `base` means no mask
  if base.size == 0: extra
  else: base +. extra

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
  mask: Tensor[T] = default(Tensor[T]),
  context: Variable[Tensor[T]] = nil,
  is_causal: bool = false,
  key_mask: Tensor[bool] = default(Tensor[bool]),
  rope: RotaryFreqs[T] = default(RotaryFreqs[T]),
  past: KVCache[T]
): tuple[output: Variable[Tensor[T]], present: KVCache[T]] =
  ## Incremental forward. Self-attention appends fresh keys/values to `past`;
  ## cross-attention caches the context projection once.
  ## Fresh keys are rotated by `rope`, cached keys keep theirs.
  ## A non-empty `past` detaches the key/value path (inference only).
  ## `mask` is additive and broadcasts from `[key]`, `[seq, key]`,
  ## `[heads, seq, key]` or `[batch, heads, seq, key]`; `key_mask` is a
  ## boolean `[batch, key]` where `true` masks the key out.
  ## Causal masking applies to self-attention only.

  let (b, n, d) = (x.value.shape[0], x.value.shape[1], self.head_dim)
  let cross = not context.isNil
  let has_rope = not rope.isEmpty
  let cached = if past.isEmpty: 0 else: past.k.shape[2]

  if cross and is_causal:
    raise newException(ValueError, "MultiHeadAttention cannot be causal in cross-attention mode")
  if not cross and self.context_dim != self.embed_dim:
    raise newException(ValueError,
      "MultiHeadAttention self-attention requires context_dim == embed_dim, got " &
      $self.context_dim & " and " & $self.embed_dim)

  # [batch, seq, heads * head_dim] <-> [batch, heads, seq, head_dim]
  template to_heads(t: Variable[Tensor[T]], heads: int): Variable[Tensor[T]] =
    t.reshape(t.value.shape[0], t.value.shape[1], heads, d).permute(0, 2, 1, 3)
  template merge_heads(t: Variable[Tensor[T]]): Variable[Tensor[T]] =
    t.permute(0, 2, 1, 3).reshape(b, n, self.dim_inner)

  var q = self.q_proj.forward(x).to_heads(self.num_heads)
  if has_rope:
    q = apply_rotary(q, rope)

  # cross-attention reuses its cached context projection
  var k, v: Variable[Tensor[T]]
  if cross and cached > 0:
    k = x.context.variable(past.k)
    v = x.context.variable(past.v)
  else:
    let src = if cross: context else: x
    k = self.k_proj.forward(src).to_heads(self.kv_heads)
    v = self.v_proj.forward(src).to_heads(self.kv_heads)
    if has_rope:
      k = apply_rotary(k, rope)
    # self-attention continues from the rotated cached prefix
    if not cross and cached > 0:
      k = x.context.variable(concat(past.k, k.value, axis = 2))
      v = x.context.variable(concat(past.v, v.value, axis = 2))

  # causal mask, offset by the number of cached keys
  var m = promote_mask(mask)
  if is_causal:
    m = add_mask(m, causal_mask[T](n, k.value.shape[2], cached))
  if key_mask.size > 0:
    m = add_mask(m, key_mask_additive[T](key_mask, -1e9.T))

  let repeats = self.num_heads div self.kv_heads
  var output = scaled_dot_product_attention(q, repeat_kv(k, repeats), repeat_kv(v, repeats), mask = m)
  if key_mask.size > 0:
    # fully masked queries attend nothing
    output = output *. x.context.variable(key_visibility[T](key_mask))

  result.output = self.out_proj.forward(output.merge_heads)
  # self-attention counts queries, cross-attention counts context tokens once
  let n_seen = if cross and cached > 0: 0
               elif cross: context.value.shape[1]
               else: n
  result.present = KVCache[T](seen: past.seen + n_seen, k: k.value, v: v.value)

proc forward*[T](
  self: MultiHeadAttention[T],
  x: Variable[Tensor[T]],
  mask: Tensor[T] = default(Tensor[T]),
  context: Variable[Tensor[T]] = nil,
  is_causal: bool = false,
  key_mask: Tensor[bool] = default(Tensor[bool]),
  rope: RotaryFreqs[T] = default(RotaryFreqs[T])
): Variable[Tensor[T]] =
  self.forward(x, mask, context, is_causal, key_mask, rope, default(KVCache[T])).output
