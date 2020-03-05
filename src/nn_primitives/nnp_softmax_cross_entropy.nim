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

import  ../tensor/backend/openmp,
        ../tensor/tensor,
        ./private/p_nnp_checks,
        ./private/p_logsumexp

# Fused numerically stable softmax + cross-entropy loss function


proc softmax_cross_entropy*[T](input, target: Tensor[T]): T =
  ## Softmax function + Cross-Entropy loss fused in one layer.
  ##
  ## Input:
  ##   - A Tensor of shape [batch_size, predicted_labels_probabilities]
  ##   - The target values of shape [batchsize, truth_labels_probability]
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

  let batch_size = input.shape[0]
  # See at the bottom of the file for explanation/proof
  result = frobenius_inner_prod(input, target)

  let sum_logsumexp = fold_axis_inline(input, T, fold_axis=0) do:
    x = y.logsumexp
  do:
    x += y.logsumexp
  do:
    x += y

  result = (sum_logsumexp - result) / T(batch_size)

proc sparse_softmax_cross_entropy*[T; Idx: SomeNumber or byte or char or enum](
        input: Tensor[T],
        target: Tensor[Idx]): T =
  ## Softmax function + Cross-Entropy loss fused in one layer.
  ##
  ## Input:
  ##   - A Tensor of shape [batchsize, predicted_labels_probabilities]
  ##   - The target values of shape [batchsize] containing the truth label id
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

  let batch_size = input.shape[0]

  # TODO proper check
  assert batch_size == target.shape[0]

  # See at the bottom of the file for explanation/proof
  # ∑i(- ti * yi) is either -yi or 0 in the sparse case.
  # Since target holds coordinates: ∑i(- ti * yi) = - yi[ti]
  {.push stacktrace:off.}
  {.push linedir:off.}
  for i in 0||(input.shape[0]-1):
    let lse = input[i,_].logsumexp

    when not declared(openmp):
      result += lse - input[i, int(target[i])]
    else:
      let tmp = lse - input[i, int(target[i])]
      # The new line is intentional or Nim inserts its frame on the line of the omp pragma
      {.emit:"""

      #pragma omp atomic
      `result` += `tmp`;""".}
  {.pop.}
  {.pop.}

  result /= T(batch_size)

# ############
# Backward pass

# TODO: optimize, slice assignment is probably very slow cf benchmark of sparse_softmax_crossentropy1
# Note: bench rowMajor/colMajor to do

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
  ##   - Both the cache and target shape should be [batchsize, features] i.e. number of samples as first dimension
  # TODO: add a `batch_axis` parameter

  let batch_size = cached_tensor.shape[0]

  # Deal with scalar and tensor gradient
  when gradient is T:
    let grad = gradient
  elif gradient is Tensor:
    let grad = gradient.unsafe_raw_offset()

  result = zeros_like(cached_tensor)

  for i in 0||(batch_size-1):
    let (max, sumexp) = cached_tensor[i,_].streaming_max_sumexp

    var res_slice = result[i,_]

    apply3_inline(res_slice, cached_tensor[i,_], target[i,_]):
      grad * (stable_softmax(y, max, sumexp) - z) / T(batch_size)


proc sparse_softmax_cross_entropy_backward*[T; Idx: SomeNumber or byte or char or enum](
        gradient: Tensor[T] or T,
        cached_tensor: Tensor[T],
        target: Tensor[Idx]
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

  # Deal with scalar and tensor gradient
  when gradient is T:
    let grad = gradient
  elif gradient is Tensor:
    let grad = gradient.unsafe_raw_offset()[0]

  let batch_size = cached_tensor.shape[0]

  result = zeros_like(cached_tensor)
  # With sparse target grad * (softmax - y) becomes:
  #   - "grad * (softmax - 1)" for the truth labels
  #   - "grad * softmax for the wrong labels
  for i, truth_idx in enumerate(target):
    result[i, int(truth_idx)] = -1

  for i in 0||(batch_size-1):
    let (max, sumexp) = cached_tensor[i, _].streaming_max_sumexp

    var res_slice = result[i, _]

    apply2_inline(res_slice, cached_tensor[i, _]):
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

# SCEi(yi, ti') = - ti' * ln( exp(yi) / ∑j exp(yj) )
#               = - ti' * (ln(exp(yi)) - (-ln ∑j exp(yj))
#               = ti' * ( ln ∑j exp(yj) - yi )

# Since we pass a minibatch of several samples we should average by minibatch size (1/batchsize)
# to keep the gradient magnitude/weight updates on the same scale as a single sample pass

# SCE(y, t') = 1/n ∑i(- ti * yi + ti * ln ∑j exp(yj))
# SCE(y, t') = 1/n [ ∑i(- ti * yi) + ∑i(ti * ln ∑j exp(yj)) ]

# ∑i(- ti * yi) is the negative dot product between 2 vectors
# ∑i(ti * ln ∑j exp(yj)) = ln ∑j exp(yj) as `ti` is a probability distribution
# and must sum to 1

# Over a batch n we have the batched_softmax_cross_entropy =
# BSCE(y, t') = 1/n ∑n dot(-tn, yn) + ∑n ln ∑j exp(yj)
# ∑n dot(-tn, yn), the generalization of dot product to matrices is
# also called the Frobenius inner product
