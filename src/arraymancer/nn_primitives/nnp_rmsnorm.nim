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

import ../tensor, math

proc rmsnorm*[T: SomeFloat](
      x, weight: Tensor[T],
      eps: T = 1e-5.T
    ): Tensor[T] =
  ## y = x / sqrt(mean(x²) + eps) * weight, x: [..., D], weight: [D]
  let d = x.shape[^1]
  assert weight.shape == [d].toMetadata, "weight shape must be [D] matching input"

  let n = x.size div d
  let xc = if x.is_C_contiguous: x else: x.clone()
  let wc = if weight.is_C_contiguous: weight else: weight.clone()

  result = newTensorUninit[T](x.shape)
  let (xb, wb, ob) = (xc.unsafe_raw_offset(), wc.unsafe_raw_offset(), result.unsafe_raw_offset())

  for r in 0 ..< n:
    let off = r * d
    var sumSq = 0.T
    for c in 0 ..< d:
      let v = xb[off + c]
      sumSq += v * v
    let invRms = 1.T / sqrt(sumSq / d.T + eps)
    for c in 0 ..< d:
      ob[off + c] = xb[off + c] * invRms * wb[c]

proc rmsnorm_backward*[T: SomeFloat](
      gradOutput, x, weight: Tensor[T],
      gradInput, gradWeight: var Tensor[T],
      eps: T = 1e-5.T
    ) =
  let d = x.shape[^1]
  assert gradOutput.shape == x.shape, "gradOutput shape must match input"
  assert weight.shape == [d].toMetadata, "weight shape must be [D] matching input"

  let n = x.size div d
  let xc = if x.is_C_contiguous: x else: x.clone()
  let wc = if weight.is_C_contiguous: weight else: weight.clone()
  let goc = if gradOutput.is_C_contiguous: gradOutput else: gradOutput.clone()

  gradInput = newTensorUninit[T](x.shape)
  gradWeight = zeros[T]([d])

  let (xb, wb, gob) = (xc.unsafe_raw_offset(), wc.unsafe_raw_offset(), goc.unsafe_raw_offset())
  let (gib, gwb) = (gradInput.unsafe_raw_offset(), gradWeight.unsafe_raw_offset())

  for r in 0 ..< n:
    let off = r * d
    var sumSq = 0.T
    for c in 0 ..< d:
      let v = xb[off + c]
      sumSq += v * v
    let invRms = 1.T / sqrt(sumSq / d.T + eps)

    var sumDxHatXHat = 0.T
    for c in 0 ..< d:
      let xHat = xb[off + c] * invRms
      let dxHat = gob[off + c] * wb[c]
      sumDxHatXHat += dxHat * xHat
      gwb[c] += gob[off + c] * xHat

    let s = sumDxHatXHat / d.T
    for c in 0 ..< d:
      let xHat = xb[off + c] * invRms
      let dxHat = gob[off + c] * wb[c]
      gib[off + c] = (dxHat - xHat * s) * invRms
