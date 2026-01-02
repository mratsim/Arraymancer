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
        ../tensor,
        ./private/p_nnp_checks,
        ./private/p_logsumexp,
        std/math

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
  if batch_size == 0:
    return 0.T

  let features = input.shape[1]

  let inp_ptr = input.unsafe_raw_buf()
  let tgt_ptr = target.unsafe_raw_buf()
  let inp_off = input.offset
  let tgt_off = target.offset
  let inp_s0 = input.strides[0]
  let inp_s1 = input.strides[1]
  let tgt_s0 = target.strides[0]
  let tgt_s1 = target.strides[1]

  {.push stacktrace:off, checks:off.}
  for i in 0||(batch_size-1):
    let row_inp_idx = inp_off + i * inp_s0
    let row_tgt_idx = tgt_off + i * tgt_s0

    var max_val = -T(Inf)
    var sum_exp: T = 0
    var row_dot: T = 0

    for j in 0 ..< features:
      let val = inp_ptr[row_inp_idx + j * inp_s1]
      let t_val = tgt_ptr[row_tgt_idx + j * tgt_s1]
      
      row_dot += val * t_val

      if val <= max_val:
        sum_exp += exp(val - max_val)
      else:
        sum_exp = sum_exp * exp(max_val - val) + 1.T
        max_val = val

    let local_loss = ln(sum_exp) + max_val - row_dot

    when declared(openmp):
      {.emit:"""
      #pragma omp atomic
      `result` += `local_loss`;""".}
    else:
      result += local_loss
  {.pop.}

  result /= T(batch_size)

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
  if batch_size == 0:
    return 0.T

  let features = input.shape[1]

  # TODO proper check
  assert batch_size == target.shape[0]

  let inp_ptr = input.unsafe_raw_buf()
  let tgt_ptr = target.unsafe_raw_buf()
  let inp_off = input.offset
  let tgt_off = target.offset
  let inp_s0 = input.strides[0]
  let inp_s1 = input.strides[1]
  let tgt_s0 = target.strides[0]

  # See at the bottom of the file for explanation/proof
  # ∑i(- ti * yi) is either -yi or 0 in the sparse case.
  # Since target holds coordinates: ∑i(- ti * yi) = - yi[ti]
  {.push stacktrace:off, checks:off.}
  for i in 0||(batch_size-1):
    let row_inp_idx = inp_off + i * inp_s0
    let row_tgt_idx = tgt_off + i * tgt_s0
    let label = int(tgt_ptr[row_tgt_idx])

    var max_val = -T(Inf)
    var sum_exp: T = 0

    for j in 0 ..< features:
      let val = inp_ptr[row_inp_idx + j * inp_s1]
      
      if val <= max_val:
        sum_exp += exp(val - max_val)
      else:
        sum_exp = sum_exp * exp(max_val - val) + 1.T
        max_val = val

    let lse = ln(sum_exp) + max_val
    let val_at_target = inp_ptr[row_inp_idx + label * inp_s1]
    let local_loss = lse - val_at_target

    when declared(openmp):
      {.emit:"""
      #pragma omp atomic
      `result` += `local_loss`;""".}
    else:
      result += local_loss
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
        ): Tensor[T] {.noinit.}=
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
        ): Tensor[T] {.noinit.}=
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
