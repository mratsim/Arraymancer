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
       math

type RotaryEmbedding*[T] = object
  ## Rotary position embedding, owned by the model and applied by attention.
  head_dim*: int
  rotary_dim*: int # 0 = full head_dim
  base*: T
  inv_freq*: Tensor[T] # [head_dim div 2], zero past rotary_dim

type RotaryFreqs*[T] = object
  ## Frequencies for a contiguous range of positions.
  cos*: Tensor[T] # [1, seq_len, head_dim div 2]
  sin*: Tensor[T]

proc init*[T: SomeFloat](
  layerType: typedesc[RotaryEmbedding[T]],
  head_dim: int,
  rotary_dim = 0,
  base = 10000.T
): RotaryEmbedding[T] =
  let rd = if rotary_dim == 0: head_dim else: rotary_dim
  doAssert head_dim > 0 and head_dim mod 2 == 0, "head_dim must be even, got " & $head_dim
  doAssert rd > 0 and rd <= head_dim and rd mod 2 == 0,
    "rotary_dim must be even and in 1 .. head_dim, got " & $rd
  result.head_dim = head_dim
  result.rotary_dim = rotary_dim
  result.base = base
  result.inv_freq = zeros[T]([head_dim div 2])
  # pairs past rotary_dim/2 keep a zero frequency: cos = 1, sin = 0
  for i in 0 ..< rd div 2:
    result.inv_freq[i] = pow(base, T(-2 * i) / T(rd))

proc isEmpty*[T](freqs: RotaryFreqs[T]): bool =
  freqs.cos.size == 0

proc forward*[T: SomeFloat](
  self: RotaryEmbedding[T],
  seq_len: int,
  offset = 0
): RotaryFreqs[T] =
  ## Frequencies for the absolute positions `offset ..< offset + seq_len`.
  let pos = arange(offset, offset + seq_len, 1).asType(T).unsqueeze(1) # [seq_len, 1]
  let angles = pos *. self.inv_freq.unsqueeze(0) # [seq_len, head_dim div 2]
  result.cos = angles.map(cos).unsqueeze(0) # [1, seq_len, head_dim div 2]
  result.sin = angles.map(sin).unsqueeze(0)

proc apply_rotary*[T](
  x: Variable[Tensor[T]],
  freqs: RotaryFreqs[T]
): Variable[Tensor[T]] =
  ## Rotate the interleaved pairs (x[2i], x[2i+1]) of a `[..., seq, head_dim]`
  ## tensor, with `freqs` covering exactly `seq` positions.
  let shape = x.value.shape
  doAssert shape.len >= 2, "apply_rotary expects at least 2 dimensions"
  let (n, d) = (shape[^2], shape[^1])
  doAssert freqs.cos.shape[1] == n,
    "rotary frequencies cover " & $freqs.cos.shape[1] & " positions, expected " & $n
  doAssert freqs.cos.shape[2] == d div 2,
    "rotary head_dim mismatch: frequencies cover " & $(freqs.cos.shape[2] * 2) &
    " channels, the tensor has " & $d
  let leading = x.value.size div (n * d)

  let cos_t = x.context.variable(freqs.cos)
  let sin_t = x.context.variable(freqs.sin)

  # rotate interleaved pairs
  let parts = x.reshape(leading, n, d div 2, 2).chunk(2, axis = 3)
  let x1 = parts[0].squeeze(3)
  let x2 = parts[1].squeeze(3)
  let y1 = x1 *. cos_t - x2 *. sin_t
  let y2 = x1 *. sin_t + x2 *. cos_t
  result = stack(y1, y2, axis = 3).reshape(shape)
