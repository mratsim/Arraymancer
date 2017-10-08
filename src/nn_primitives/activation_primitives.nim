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

import  ../tensor/tensor,
        math

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

# TODO: tests

proc sigmoid*[T: SomeReal](t: Tensor[T]): Tensor[T] {.inline.}=
  ## Logistic sigmoid activation function, :math:`f(x) = 1 / (1 + \exp(-x))`
  ## Note: Canonical sigmoid is not stable for large negative value

  # stable: proc sigmoid_closure(x: T): T = 0.5.T * (tanh(0.5.T * x) + 1.T)

  return t.mapT(1.T / (1.T + exp(-x)))

proc msigmoid*[T: SomeReal](t: var Tensor[T]) {.inline.}=
  ## Logistic sigmoid activation function, :math:`f(x) = 1 / (1 + \exp(-x))`
  ## Note: Canonical sigmoid is not stable for large negative value

  # stable: proc sigmoid_closure(x: T): T = 0.5.T * (tanh(0.5.T * x) + 1.T)
  t.applyT(1.T / (1.T + exp(-x)))

proc relu*[T](t: Tensor[T]): Tensor[T] {.inline.}=
  t.mapT max(0.T,x)

proc mrelu*[T](t: var Tensor[T]) {.inline.}=
  t.applyT max(0.T, x)


proc relu_backward*[T](gradient: Tensor[T], cached_tensor: Tensor[T]): Tensor[T]{.inline.}=
  result = cached_tensor.mapT(if x <= 0.T: 0.T else: 1.T)
  result .*= gradient

proc sigmoid_backward*[T](gradient: Tensor[T], cached_tensor: Tensor[T]): Tensor[T]{.inline.}=
  result = cached_tensor.mapT(x * (1 - x))
  result .*= gradient