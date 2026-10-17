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

import ../tensor,
       ./nnp_softmax,
       math

proc causal_mask*[T: SomeFloat](
    n_query, n_key: int,
    offset: int,
    mask_val: T = -1e9.T
): Tensor[T] =
  ## Causal mask where query i is at absolute position offset + i
  ## and may only attend keys j <= offset + i.
  result = zeros[T]([1, 1, n_query, n_key])
  for i in 0 ..< n_query:
    for j in (offset + i + 1) ..< n_key:
      result[0, 0, i, j] = mask_val

proc causal_mask*[T: SomeFloat](n: int, mask_val: T = -1e9.T): Tensor[T] =
  ## Square causal mask where query `i` may attend keys `j <= i`.
  causal_mask[T](n, n, 0, mask_val)

proc scaled_dot_product_attention*[T: SomeFloat](
    query, key, value: Tensor[T],
    scale: T = 0.T,
    mask: Tensor[T] = default(Tensor[T]),
    bias: Tensor[T] = default(Tensor[T])
): Tensor[T] =
  ## `softmax(query keyᵀ * scale + bias + mask) value`
  let d = query.shape[^1]
  let s = if scale == 0.T: 1.T / sqrt(d.T) else: scale

  var scores = (query * key.transpose2d) *. s
  if bias.size > 0:
    scores = scores +. bias
  if mask.size > 0:
    scores = scores +. mask

  result = scores.softmax(axis = -1) * value
