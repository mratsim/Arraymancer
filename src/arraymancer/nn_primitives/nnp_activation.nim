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
import  ../tensor,
        ./private/p_activation

# Neural net activation functions that works directly on Tensors
# TODO: tests

# ##################################################################################################
# Forward

proc sigmoid*[T: SomeFloat](t: Tensor[T]): Tensor[T] {.noInit.}=
  ## Logistic sigmoid activation function, :math:`f(x) = 1 / (1 + \exp(-x))`
  ## Note: Canonical sigmoid is not stable for large negative value
  ## Please use sigmoid_cross_entropy for the final layer for better stability and performance

  # stable: proc sigmoid_closure(x: T): T = 0.5.T * (tanh(0.5.T * x) + 1.T)

  result = newTensorUninit[T](t.shape)
  forEach dst in result, src in t:
    dst = sigmoid(src)

proc relu*[T](t: Tensor[T]): Tensor[T] {.noInit.}=
  result = newTensorUninit[T](t.shape)
  forEach dst in result, src in t:
    dst = max(0.T, src)

proc tanh*[T: SomeFloat](t: Tensor[T]): Tensor[T] {.noInit.}=
  result = newTensorUninit[T](t.shape)
  forEach dst in result, src in t:
    dst = tanh(src)

# ##################################################################################################
# In-place forward

proc msigmoid*[T: SomeFloat](t: var Tensor[T]) =
  ## Logistic sigmoid activation function, :math:`f(x) = 1 / (1 + \exp(-x))`
  ## Note: Canonical sigmoid is not stable for large negative value

  # stable: proc sigmoid_closure(x: T): T = 0.5.T * (tanh(0.5.T * x) + 1.T)
  forEach x in t:
    x = sigmoid(x)

proc mrelu*[T](t: var Tensor[T]) =
  forEach x in t:
    x = t.apply_inline max(0.T, x)

proc mtanh*[T: SomeFloat](t: var Tensor[T]) =
  forEach x in t:
    x = tanh(x)

# ##################################################################################################
# Backward

proc sigmoid_backward*[T](gradient: Tensor[T], cached_tensor: Tensor[T]): Tensor[T]{.noInit.}=
  result = newTensorUninit[T](gradient.shape)
  forEach r in result, c in cached_tensor, g in gradient:
    r = c*(1-c) * g

proc relu_backward*[T](gradient: Tensor[T], cached_tensor: Tensor[T]): Tensor[T]{.noInit.}=
  result = newTensorUninit[T](gradient.shape)
  forEach r in result, c in cached_tensor, g in gradient:
    if c <= 0.T:
      r = 0.T
    else:
      r = g

proc tanh_backward*[T](gradient: Tensor[T], cached_tensor: Tensor[T]): Tensor[T]{.noInit.}=
  result = newTensorUninit[T](gradient.shape)
  forEach r in result, c in cached_tensor, g in gradient:
    r = g * (1 - c * c)

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
