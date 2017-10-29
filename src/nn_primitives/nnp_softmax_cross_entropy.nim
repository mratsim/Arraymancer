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
        ./private/p_nnp_checks,
        ./private/p_logsumexp,
        math

# Fused numerically stable softmax + cross-entropy loss function


proc softmax_cross_entropy*[T](input, target: Tensor[T]): T {.inline.} =
  ## Softmax function + Cross-Entropy loss fused in one layer.
  ## This leverage the log-sum-exp trick for improved numerical stability
  ## It is also faster than calling both separately
  ##
  ## Input:
  ##   - A Tensor
  ##   - The target values
  ## Returns:
  ##   - Apply a softmax activation and returns the cross-entropy loss.
  ## Shape:
  ##   - Both the cache and target shape should be @[features, batchsize] i.e. number of samples as last dimension
  # TODO: add a `batch_axis` parameter


  # TODO: term rewriting macro for auto fusion

  when compileOption("boundChecks"):
    check_input_target(input, target)

  # We need parallel fused:
  #   fold_axis (log softmax per sample)
  #   -> map2 (cross-entropy)
  #   -> reduce (sum) for all loss functions

  # 1. Create a temporary tensor with the crossentropy per sample
  var sample_softmax_xentropy = zeros[T](1, input.shape[1])
  var i = 0
  for sample_input, sample_target in zipAxis(input, target, 1):
    sample_softmax_xentropy[0, i] = sum:
      map2_inline(sample_input, sample_target):
        # SCEi(yi, ti') = ti * ( ln ∑j exp(yij) - yij ) # see below
        y * (sample_input.logsumexp - x)
    inc i

  # 2. Sum the sample crossentropies
  result = sample_softmax_xentropy.sum


# ################################################
# Explanation of softmax cross-entropy algorithms:

# ############
# Forward pass

# Cross-entropy has the following form (ti' being the truth/target probabilities and yi the predicted probabilities)
# i is the row index (batch id)
# j is the column index (label id)
# For a single row of labels
# CEi(yi, ti') = − ti' ln(yi)

# Softmax has the form:
# Softmax(yj) = exp(yj) / ∑j exp(yj)

# SCEi(yi, ti') = - ti' * ln( exp(yij) / ∑j exp(yij) )
#               = - ti' * (ln(exp(yij)) - (-ln ∑j exp(yij))
#               = ti' * ( ln ∑j exp(yij) - yij )

# Now by considering the whole batch
# Since we pass a minibatch of several samples we should average by minibatch size (1/batchsize)
# to keep the gradient magnitude/weight updates on the same scale as a single sample pass

# SCE(y, t') = 1/n ∑i(- ti * yij + ln ∑j exp(yij))


# ############
# Backward pass