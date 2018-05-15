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

import math
import  ../tensor/tensor,
        ./private/p_activation,
        ./private/p_logsumexp

# Neural net activation functions that works directly on Tensors
# TODO: tests

# ##################################################################################################
# Forward

proc sigmoid*[T: SomeReal](t: Tensor[T]): Tensor[T] {.noInit.}=
  ## Logistic sigmoid activation function, :math:`f(x) = 1 / (1 + \exp(-x))`
  ## Note: Canonical sigmoid is not stable for large negative value
  ## Please use sigmoid_cross_entropy for the final layer for better stability and performance

  # stable: proc sigmoid_closure(x: T): T = 0.5.T * (tanh(0.5.T * x) + 1.T)

  result = map_inline(t):
    sigmoid(x)

proc relu*[T](t: Tensor[T]): Tensor[T] {.noInit.}=
  t.map_inline max(0.T,x)

proc tanh*[T: SomeReal](t: Tensor[T]): Tensor[T] {.noInit.}=
  t.map_inline tanh(x)

# ##################################################################################################
# In-place forward

proc msigmoid*[T: SomeReal](t: var Tensor[T]) =
  ## Logistic sigmoid activation function, :math:`f(x) = 1 / (1 + \exp(-x))`
  ## Note: Canonical sigmoid is not stable for large negative value

  # stable: proc sigmoid_closure(x: T): T = 0.5.T * (tanh(0.5.T * x) + 1.T)
  apply_inline(t):
    sigmoid(x)

proc mrelu*[T](t: var Tensor[T]) =
  t.apply_inline max(0.T, x)

proc mtanh*[T: SomeReal](t: var Tensor[T]) =
  t.apply_inline tanh(x)

# ##################################################################################################
# Backward

proc sigmoid_backward*[T](gradient: Tensor[T], cached_tensor: Tensor[T]): Tensor[T]{.noInit.}=
  result = map2_inline(cached_tensor, gradient):
    x * (1 - x) * y

proc relu_backward*[T](gradient: Tensor[T], cached_tensor: Tensor[T]): Tensor[T]{.noInit.}=
  result = map2_inline(cached_tensor, gradient):
    if x <= 0.T:
      0.T
    else:
      y

proc tanh_backward*[T](gradient: Tensor[T], cached_tensor: Tensor[T]): Tensor[T]{.noInit.}=
  result = map2_inline(cached_tensor, gradient):
    y * (1 - x * x)

# ####################################################################################################
# Documentation

# Sigmoid implementation doc:
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
