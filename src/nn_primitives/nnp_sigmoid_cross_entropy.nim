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

import  ../math_ops_fusion/math_ops_fusion,
        ../tensor/tensor,
        ./private/p_nnp_checks,
        ./private/p_activation,
        math

# Fused numerically stable sigmoid + cross-entropy loss function

proc sigmoid_cross_entropy*[T](input, target: Tensor[T]): T =
  ## Sigmoid function + Cross-Entropy loss fused in one layer.
  ##
  ## Input:
  ##   - A Tensor
  ##   - The target values
  ## Returns:
  ##   - Apply a sigmoid activation and returns the cross-entropy loss.
  ## Shape:
  ##   - Both the cache and target shape should be [batch_size, features] i.e. number of samples as first dimension
  # TODO: add a `batch_axis` parameter

  # TODO: term rewriting macro for auto fusion

  when compileOption("boundChecks"):
    check_input_target(input, target)

  let batch_size = input.shape[0]

  # ln1p(x) does ln(1 + x) but avoids catastrophic cancellation if x << 1.

  # result = 0.T
  # for xi, ti in zip(input, target):
  #   result += (-ti * xi +  max(xi,0) + ln1p(exp(-abs(xi))) ) / T(input.shape[1])

  # We need parallel fused map2 -> reduce for all loss functions
  result = sum:
    map2_inline(input, target):
      -y * x +  max(x,0) + ln1p(exp(-abs(x))) # This leverage the logsumexp trick to improve numerical stability

  # Normalize by batch_size
  result /= T(batch_size)

proc sigmoid_cross_entropy_backward*[T](
        gradient: Tensor[T] or T,
        cached_tensor: Tensor[T],
        target: Tensor[T]
        ): Tensor[T] {.noInit.}=
  ## Derivatives of sigmoid_cross_entropy
  ## Input:
  ##   - The input gradient as a scalar or a Tensor
  ##   - A cache tensor that contains data from before the forward pass
  ##   - The target values
  ## Shape:
  ##   - Both the cache and target shape should be [batch_size, features] i.e. number of samples as first dimension
  # TODO: add a `batch_axis` parameter

  let batch_size = cached_tensor.shape[0]

  # Deal with scalar and tensor gradient
  when gradient is T:
    let grad = gradient
  elif gradient is Tensor:
    let grad = gradient.unsafe_raw_offset()[0]

  result = map2_inline(cached_tensor, target):
    grad * (sigmoid(x) - y) / T(batch_size)

# ################################################
# Explanation of sigmoid cross-entropy algorithms:

# ############
# Forward pass

# Binary cross-entropy has the following form for a single sample (ti' being the truth/target probabilities and yi the predicted probabilities)
# BCEi(yi, ti') = − ( ti' ln(yi) + (1−ti') ln(1−yi) )

# Since we pass a minibatch of several samples we should average by minibatch size (1/batchsize)
# to keep the gradient magnitude/weight updates on the same scale as a single sample pass
# BCE(y, t') = − 1/n ∑i( ti' ln(yi) + (1−ti') ln(1−yi) )

# ln(yi) = ln(sigmoid(xi)) = ln(1/(1+e^-xi)) = ln(e^xi/( 1 + e^xi ))
# ln(yi) = x - ln(1 + e^xi)

# ln(1 - yi) = ln(1 - sigmoid(xi)) = ln(1 + e^xi - e^xi) / (1 + e^xi))
# ln(1 - yi) = - ln(1 + e^xi)

# Replacing Sigmoid Cross Entropy
# SCE(x, t') = − 1/n ∑i(ti' * (xi - ln(1 + e^xi)) + (1−ti') * -ln(1 + e^xi) )
#            = − 1/n ∑i(ti' * xi - ti' * ln(1 + e^xi) -ln(1 + e^xi) + ti' * ln(1 + e^xi) )
#            = − 1/n ∑i(ti' * xi - ln(1 + e^xi) )
#            = − 1/n ∑i(ti' * xi - ln(e^0 + e^xi) )
#
# Using the logsumexp trick with factorize by a constant to improve numerical stability
# c = max(xi, 0)
#
# SCE(x, t') = − 1/n ∑i(ti' * xi - ln(e^c *( e^(0-c) + e^(xi-c))
#            = − 1/n ∑i(ti' * xi - ln(e^c *( e^(0-c) + e^(xi-c))
#            = − 1/n ∑i(ti' * xi - c - ln(e^-c + e^(xi-c))
#
# If c = xi (xi > 0), ln(e^-c + e^(xi-c)) becomes ln(e^-xi + 1)
# else c = 0 (xi < 0 ), ln(e^-c + e^(xi-c)) becomes ln(1 + e^xi)
# Both cases are covered by ln(1 + e^-|xi|)
#
# Finally
# SCE(x, t') = − 1/n ∑i(ti' * xi - max(xi,0) - ln(1 + e^-|xi|)
#
#
#
# Other idea: streaming maximum (http://www.nowozin.net/sebastian/blog/streaming-log-sum-exp-computation.html)
#

# #############
# Backward pass

# Derivative of Sigmoid-CE:
# We start from this formula: SCE(x, t') = − 1/n ∑i(ti' * xi - ln(1 + e^xi) )
#                                        = 1/n ∑i(-ti' * xi + ln(1 + e^xi) )
#
# On a single sample:
# dSCE/dxi = d/dxi (-ti' * xi + ln(1 + e^xi))
#          = -ti' + e^xi * 1/(1 + e^xi))
#          = -ti' + sigmoid(xi)
#
# For a vector of samples
# dSCE/dx = 1/n ∑i( sigmoid(xi) - ti' )
