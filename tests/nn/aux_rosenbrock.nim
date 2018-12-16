# Copyright 2017-Present Mamy André-Ratsimbazafy & the Arraymancer contributors
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

# We use the Rosenbrock function for testing optimizers
# https://en.wikipedia.org/wiki/Rosenbrock_function

func rosenbrock*[T](x, y: Tensor[T], a: static T = 1, b: static T = 100): Tensor[T] =
  # f(x, y) = (a - x)² + b(y - x²)²
  result = map2_inline(x, y):
    let u = a - x
    let v = y - x*x
    u * u + b * v * v

block:
  let x = [1].toTensor
  let y = [1].toTensor
  doAssert rosenbrock(x, y) == [0].toTensor

block:
  let x = [5].toTensor
  let y = [25].toTensor
  doAssert rosenbrock(x, y, 5, 999) == [0].toTensor

func drosenbrock*[T](x, y: Tensor[T], a: static T = 1, b: static T = 100): tuple[dx, dy: Tensor[T]] =
  result.dx = map2_inline(x, y):
    2.T*x - 2.T*a + 4*b*x*x*x - 4*b*x*y
  result.dy = map2_inline(x, y):
    2*b*(y - x*x)

block:
  let x = [1].toTensor
  let y = [1].toTensor
  doAssert drosenbrock(x, y) == ([0].toTensor, [0].toTensor)
