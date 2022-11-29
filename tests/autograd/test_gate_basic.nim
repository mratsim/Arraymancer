# Copyright 2017-2018 Mamy André-Ratsimbazafy & the Arraymancer contributors
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
import unittest, sequtils

proc main() =
  suite "Autograd of basic operations":
    test "Gradient of tensor addition":

      let a = toSeq(1..8).toTensor.reshape(2,4).astype(float32)
      let b = toSeq(11..18).toTensor.reshape(2,4).astype(float32)

      let ctx = newContext Tensor[float32]

      let va = ctx.variable(a, requires_grad = true)
      let vb = ctx.variable(b, requires_grad = true)

      let vc = va + vb

      vc.backprop()

      let onesTensor = ones[float32](2, 4)

      check: va.grad == onesTensor
      check: vb.grad == onesTensor

    test "Gradient of mean":

      let a = toSeq(1..8).toTensor.reshape(2,4).astype(float32)

      let ctx = newContext Tensor[float32]

      let va = ctx.variable(a, requires_grad = true)
      let m = va.mean()

      m.backprop()

      let constantTensor = ones[float32](2, 4) / 8.0

      check: va.grad == constantTensor

    test "Gradient of mean along one axis":

      let a = toSeq(1..8).toTensor.reshape(2,4).astype(float32)

      let ctx = newContext Tensor[float32]

      let va = ctx.variable(a, requires_grad = true)

      let m0 = va.mean(axis=0)
      m0.backprop()
      let constantTensor0 = ones[float32](2, 4) / 4.0
      check: va.grad == constantTensor0

      va.grad = zeros_like(va.grad)

      let m = va.mean(axis=1)
      m.backprop()
      let constantTensor1 = ones[float32](2, 4) / 2.0
      check: va.grad == constantTensor1

main()
GC_fullCollect()
