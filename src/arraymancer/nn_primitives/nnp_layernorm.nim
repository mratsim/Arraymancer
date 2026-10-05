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

proc layernorm*[T: SomeFloat](
      x, weight: Tensor[T],
      bias: Tensor[T] = default(Tensor[T]),
      eps: T = 1e-5.T
    ): Tensor[T] =
  ## y = (x - mean) / sqrt(var + eps) * weight + bias, x: [..., D], weight/bias: [D]
  let d = x.shape[^1]
  assert weight.shape == [d].toMetadata, "weight shape must be [D] matching input"
  let hasBias = bias.size > 0
  if hasBias:
    assert bias.shape == [d].toMetadata, "bias shape must be [D] matching input"

  let n = x.size div d
  let xc = if x.is_C_contiguous: x else: x.clone()
  let wc = if weight.is_C_contiguous: weight else: weight.clone()
  var bc = default(Tensor[T])
  if hasBias:
    bc = if bias.is_C_contiguous: bias else: bias.clone()

  result = newTensorUninit[T](x.shape)
  let (xb, wb, ob) = (xc.unsafe_raw_offset(), wc.unsafe_raw_offset(), result.unsafe_raw_offset())

  for r in 0 ..< n:
    let off = r * d
    var sumVal = 0.T
    for c in 0 ..< d:
      sumVal += xb[off + c]
    let mean = sumVal / d.T

    var sumSq = 0.T
    for c in 0 ..< d:
      let diff = xb[off + c] - mean
      sumSq += diff * diff
    let invStd = 1.T / sqrt(sumSq / d.T + eps)

    for c in 0 ..< d:
      ob[off + c] = (xb[off + c] - mean) * invStd * wb[c]

  if hasBias:
    let bb = bc.unsafe_raw_offset()
    for r in 0 ..< n:
      let off = r * d
      for c in 0 ..< d:
        ob[off + c] += bb[c]

proc layernorm_backward*[T: SomeFloat](
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
  if gradWeight.size == 0 or gradWeight.shape != [d].toMetadata:
    gradWeight = zeros[T]([d])

  let (xb, wb, gob) = (xc.unsafe_raw_offset(), wc.unsafe_raw_offset(), goc.unsafe_raw_offset())
  let (gib, gwb) = (gradInput.unsafe_raw_offset(), gradWeight.unsafe_raw_offset())

  for r in 0 ..< n:
    let off = r * d
    var sumVal = 0.T
    for c in 0 ..< d:
      sumVal += xb[off + c]
    let mean = sumVal / d.T

    var sumSq = 0.T
    for c in 0 ..< d:
      let diff = xb[off + c] - mean
      sumSq += diff * diff
    let invStd = 1.T / sqrt(sumSq / d.T + eps)

    var sumDxHat = 0.T
    var sumDxHatXHat = 0.T
    for c in 0 ..< d:
      let xHat = (xb[off + c] - mean) * invStd
      let dxHat = gob[off + c] * wb[c]
      sumDxHat += dxHat
      sumDxHatXHat += dxHat * xHat
      gwb[c] += gob[off + c] * xHat

    let meanDxHat = sumDxHat / d.T
    let meanDxHatXHat = sumDxHatXHat / d.T

    for c in 0 ..< d:
      let xHat = (xb[off + c] - mean) * invStd
      let dxHat = gob[off + c] * wb[c]
      gib[off + c] = (dxHat - meanDxHat - xHat * meanDxHatXHat) * invStd

proc layernorm_backward*[T: SomeFloat](
      gradOutput, x, weight: Tensor[T],
      gradInput, gradWeight, gradBias: var Tensor[T],
      eps: T = 1e-5.T
    ) =
  layernorm_backward(gradOutput, x, weight, gradInput, gradWeight, eps)
  let d = x.shape[^1]
  gradBias = gradOutput.reshape(gradOutput.size div d, d).sum(axis = 0).squeeze(0)
