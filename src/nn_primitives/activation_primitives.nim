# Copyright 2017 the Arraymancer contributors
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

import ../arraymancer, math

# Neural net activation functions that works directly on Tensors


# Note:
# 1. Canonical sigmoid "f(x) = 1 / (1 + exp(-x))" is unstable
# for negative values < 709 (for float64)
#
# 2. Alternative expression stable for negative but unstable for positive is
# "f(x) = exp(x) / (1 + exp(x))"
#
# 3. Introducing branching would be very costly.
#
# 4. Using tanh as 0.5 * (tanh(0.5 * x) + 1) is better than branching
# but slow as well
#
# 5. Another alternative would be to clip x to max (-500, x) to avoid this instability
#
# Benchmarks available in the benchmark folder
#

proc sigmoid*[T: SomeReal](t: Tensor[T]): Tensor[T] {.inline.}=
  ## Logistic sigmoid activation function, :math:`f(x) = 1 / (1 + \exp(-x))`
  ## Note: Canonical sigmoid is not stable for large negative value

  proc sigmoid_closure(x: T): T = 1.T / (1.T + exp(-x))

  # stable: proc sigmoid_closure(x: T): T = 0.5.T * (tanh(0.5.T * x) + 1.T)

  return t.map(sigmoid_closure)

proc msigmoid*[T: SomeReal](t: var Tensor[T]): Tensor[T] {.inline.}=
  ## Logistic sigmoid activation function, :math:`f(x) = 1 / (1 + \exp(-x))`
  ## Note: Canonical sigmoid is not stable for large negative value

  proc sigmoid_closure(x: T): T = 1.T / (1.T + exp(-x))

  # stable: proc sigmoid_closure(x: T): T = 0.5.T * (tanh(0.5.T * x) + 1.T)

  return t.map(sigmoid_closure)

proc relu*[T](t: Tensor[T]): Tensor[T] {.inline.}=
  proc relu_closure(x: T): T =
    max(0.T, x)
  t.map(relu_closure)

proc mrelu*[T](t: var Tensor[T]): Tensor[T] {.inline.}=
  proc relu_closure(x: T): T =
    max(0.T, x)
  t.apply(relu_closure)