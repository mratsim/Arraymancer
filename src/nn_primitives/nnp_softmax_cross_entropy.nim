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


proc softmax_cross_entropy*[T](input, target: Tensor[T]): T =
  ## Softmax function + Cross-Entropy loss fused in one layer.
  ##
  ## Input:
  ##   - A Tensor of shape @[predicted_labels_probabilities, batchsize]
  ##   - The target values of shape @[truth_labels_probability, batchsize]
  ## Returns:
  ##   - Apply a softmax activation and returns the cross-entropy loss.
  ##
  ## ``Softmax_cross_entropy`` measures the cross-entropy error for multiclass classification.
  ## Classes are mutually exclusive (only 1 label is true) but the truth labels (``target``) need not be.
  ##
  ## Note: Instead of one-hot-encoded labels, it is more efficient to use ``sparse_softmax_cross_entropy``
  ## instead of feeding ``softmax_cross_entropy``.
  ##
  ## For example if your true probablities are (car: 0.10, airplane: 0.60, bike: 0.05, bus: 0.25),
  ## you have to use ``softmax_cross_entropy``
  ##
  ## However if your true probablities are (car: 0, airplane: 1, bike: 0, bus: 0) (a one-hot-encoded vector),
  ## you should prefer ``sparse_softmax_cross_entropy``


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
    # SCEi(yi, ti') = ti * ( ln ∑j exp(yij) - yij ) # see below
    let lse = sample_input.logsumexp
    sample_softmax_xentropy[0, i] = sum:
      map2_inline(sample_input, sample_target):
        y * (lse - x)
    inc i

  # 2. Sum the sample crossentropies and normalize by batchsize
  # In the softmax case batchsize = sample_softmax_xentropy.size
  # So we can use the "mean" function
  result = sample_softmax_xentropy.mean

proc sparse_softmax_cross_entropy*[T](input: Tensor[T], target: Tensor[int]): T =
  ## Softmax function + Cross-Entropy loss fused in one layer.
  ##
  ## Input:
  ##   - A Tensor of shape @[predicted_labels_probabilities, batchsize]
  ##   - The target values of shape @[batchsize] containing the truth label id
  ## Returns:
  ##   - Apply a softmax activation and returns the cross-entropy loss.
  ##
  ## ``sparse_softmax_cross_entropy`` measures the cross-entropy error for multiclass classification.
  ## Classes are mutually exclusive (only 1 label is true).
  ##
  ## Important: [0, 0, 1] means label 2 is true i.e. labels start at 0
  ##
  ## Note: Instead of one-hot-encoded labels, it is more efficient to use ``sparse_softmax_cross_entropy``
  ## instead of feeding ``softmax_cross_entropy``.
  ##
  ## For example if your true probablities are (car: 0.10, airplane: 0.60, bike: 0.05, bus: 0.25),
  ## you have to use ``softmax_cross_entropy``
  ##
  ## However if your true probablities are (car: 0, airplane: 1, bike: 0, bus: 0) (a one-hot-encoded vector),
  ## you should prefer ``sparse_softmax_cross_entropy``


  # TODO: term rewriting macro for auto fusion

  # TODO proper check
  assert input.shape[1] == target.shape[0]

  # We need parallel mapAxis_inline

  # 1. Create a temporary tensor with the crossentropy per sample
  var sample_softmax_xentropy = zeros[T](1, input.shape[1])

  var i = 0
  for sample_input, sample_target in zipAxis(input, target.unsafeUnsqueeze(0), 1):
    # SCEi(yi, ti') = ti * ( ln ∑j exp(yij) - yij )
    # ti is 1 or 0 since labels are sparse
    # So we can simplify to SCEi(yi, ti') = ln ∑j exp(yij) - yi[ti] i.e. use target label id as index
    # While iterating on the axis ``ti`` is sample_target[0]
    let lse = sample_input.logsumexp
    sample_softmax_xentropy[0, i] = lse - sample_input[sample_target[0,0], 0]
    inc i

  # 2. Sum the sample crossentropies and normalize by batchsize
  result = sample_softmax_xentropy.mean

# ############
# Backward pass

proc stable_softmax[T](x, max, sumexp: T): T {.noSideEffect, inline.}=
  # Numerically stable streaming softmax helper
  result = exp(x - max) / sumexp

proc softmax_cross_entropy_backward*[T](
        gradient: Tensor[T] or T,
        cached_tensor: Tensor[T],
        target: Tensor[T]
        ): Tensor[T] {.noInit.}=
  ## Derivatives of softmax_cross_entropy
  ## Input:
  ##   - The input gradient as a scalar or a Tensor
  ##   - A cache tensor that contains data from before the forward pass
  ##   - The target values
  ## Shape:
  ##   - Both the cache and target shape should be [features, batchsize] i.e. number of samples as last dimension
  # TODO: add a `batch_axis` parameter

  let batch_size = cached_tensor.shape[^1]

  # Deal with scalar and tensor gradient
  when gradient is T:
    let grad = gradient
  elif gradient is Tensor:
    let grad = gradient.data[gradient.offset]

  result = zeros_like(cached_tensor)

  var sample_id = 0
  for truth_idx in axis(target, 1):
    let (max, sumexp) = cached_tensor[_,sample_id].streaming_max_sumexp

    result[_,sample_id] = map2_inline(cached_tensor[_,sample_id], target[_,sample_id]):
      grad * (stable_softmax(x, max, sumexp) - y) / T(batch_size)

    inc sample_id

proc sparse_softmax_cross_entropy_backward*[T](
        gradient: Tensor[T] or T,
        cached_tensor: Tensor[T],
        target: Tensor[int]
        ): Tensor[T] {.noInit.}=
  ## Derivatives of sparse_softmax_cross_entropy
  ## Input:
  ##   - The input gradient as a scalar or a Tensor
  ##   - A cache tensor that contains data from before the forward pass
  ##   - The target values
  ## Shape:
  ##   - Both the cache should be [features, batchsize] i.e. number of samples as last dimension
  ##   - target shape should be [batchsize]
  # TODO: add a `batch_axis` parameter

  let batch_size = cached_tensor.shape[^1]

  # Deal with scalar and tensor gradient
  when gradient is T:
    let grad = gradient
  elif gradient is Tensor:
    let grad = gradient.data[gradient.offset]

  result = zeros_like(cached_tensor)
  # With sparse target grad * (softmax - y) becomes:
  #   - "grad * (softmax - 1)" for the truth labels
  #   - "grad * softmax for the wrong labels
  for sample_id, truth_idx in enumerate(target):
    result[truth_idx, sample_id] = -1

  for sample_id, truth_idx in enumerate(target):
    let (max, sumexp) = cached_tensor[_,sample_id].streaming_max_sumexp

    # We can't directly use apply2_inline on result[_, sample_id]
    # due to https://github.com/mratsim/Arraymancer/issues/52
    var res_slice = result.unsafeSlice(_, sample_id)

    apply2_inline(res_slice, cached_tensor[_,sample_id]):
      grad * (stable_softmax(y, max, sumexp) + x) / T(batch_size)


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