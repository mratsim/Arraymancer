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

import ../../arraymancer
import math

# Sigmoid cross-entropy function that works directly on Tensors
# and provide control without autograd

proc check_input_target[T](input, target: Tensor[T]) {.inline.}=
  if input.shape != target.shape:
    raise newException(ValueError, "Input shape " & $input.shape &
      " and target shape " & $target.shape & " should be the same")

proc sigmoid_cross_entropy*[T](input, target: Tensor[T]): T {.inline.} =
  ## Sigmoid function + Cross-Entropy loss fused in one layer.
  ## This leverage the log-sum-exp trick for improved numerical stability
  ## It is also faster than calling both separately
  ##
  ## Input:
  ##   - A Tensor
  ##   - The target values
  ## Returns:
  ##   - Apply and sigmoid activation and returns the cross-entropy loss.
  ## Shape:
  ##   - Both the cache and target shape should be @[features, batchsize] i.e. number of samples as last dimension


  # TODO: term rewriting macro for auto fusion

  when compileOption("boundChecks"):
    check_input_target(input, target)

  result = 0.T
  for xi, ti in zip(input, target):
    result += (-ti * xi +  max(xi,0) + ln(1 + exp(-abs(xi))) ) / T(input.shape[1]) #input.shape[1] is the batch size


proc sigmoid_cross_entropy_backward*[T](
        gradient: Tensor[T] or T,
        cached_tensor: Tensor[T],
        target: Tensor[T]
        ): Tensor[T] {.inline.} =
  ## Derivatives of sigmoid_cross_entropy
  ## Input:
  ##   - The input gradient as a scalar or a Tensor
  ##   - A cache tensor that contains data from before the forward pass
  ##   - The target values
  ## Shape:
  ##   - Both the cache and target shape should be @[features, batchsize] i.e. number of samples as last dimension
  let batch_size = cached_tensor.shape[^1]

  # Deal with scalar and tensor gradient
  when gradient is T:
    let grad = gradient
  elif gradient is Tensor:
    let grad = gradient.data[gradient.offset]

  proc sigmoid_cross_entropy_backward_closure[T](xi, ti: T): T =
    grad * ( 1.T / (1.T + exp(-xi)) - ti) / T(batch_size)

  return map2(cached_tensor, sigmoid_cross_entropy_backward_closure, target)

# ################################################
# Explanation of sigmoid cross-entropy algorithms:

# ############
# Forward pass

# Cross-entropy has the following form for a single sample
# CEi(yi, yi') = − ( ti ln(yi) + (1−ti) ln(1−yi) )

# Since we pass a minibatch of several samples we should average by minibatch size (1/batchsize)
# to keep the gradient magnitude/weight updates on the same scale as a single sample pass
# CE(y, y') = − 1/n ∑i( ti ln(yi) + (1−ti) ln(1−yi) )

# yi = ln(sigmoid(xi)) = ln(1/(1+e^-xi)) = ln(e^xi/( 1 + e^xi ))
# yi = x - ln(1 + e^xi)

# 1 - yi = ln(1 - sigmoid(xi)) = ln(1 + e^xi - e^xi) / (1 + e^xi))
# 1 - yi = - ln(1 + e^xi)

# Replacing Sigmoid Cross Entropy
# SCE(x, y') = − 1/n ∑i(ti * (xi - ln(1 + e^xi)) + (1−ti) * -ln(1 + e^xi) )
#            = − 1/n ∑i(ti * xi - ti * ln(1 + e^xi) -ln(1 + e^xi) + ti * ln(1 + e^xi) )
#            = − 1/n ∑i(ti * xi - ln(1 + e^xi) )
#            = − 1/n ∑i(ti * xi - ln(e^0 + e^xi) )
#
# Using the logsumexp trick with factorize by a constant
# c = max(xi, 0)
#
# SCE(x, y') = − 1/n ∑i(ti * xi - ln(e^c *( e^(0-c) + e^(xi-c))
#            = − 1/n ∑i(ti * xi - ln(e^c *( e^(0-c) + e^(xi-c))
#            = − 1/n ∑i(ti * xi - c - ln(e^-c + e^(xi-c))
#
# If c = xi (xi > 0), ln(e^-c + e^(xi-c)) becomes ln(e^-xi + 1)
# else c = 0 (xi < 0 ), ln(e^-c + e^(xi-c)) becomes ln(1 + e^xi)
# Both cases are covered by ln(1 + e^-|xi|)
#
# Finally
# SCE(x, y') = − 1/n ∑i(ti * xi - max(xi,0) - ln(1 + e^-|xi|)
#
#
#
# Other idea: streaming maximum (http://www.nowozin.net/sebastian/blog/streaming-log-sum-exp-computation.html)
#

# #############
# Backward pass

# Derivative of Sigmoid-CE:
# We start from this formula: SCE(x, y') = − 1/n ∑i(ti * xi - ln(1 + e^xi) )
#                                        = 1/n ∑i(-ti * xi + ln(1 + e^xi) )
#
# On a single sample:
# dSCE/dxi = d/dxi (-ti * xi + ln(1 + e^xi))
#          = -ti + e^xi * 1/(1 + e^xi))
#          = -ti * sigmoid(xi)
#
# For a vector of samples
# dSCE/dx = 1/n ∑i( sigmoid(xi) - ti )
