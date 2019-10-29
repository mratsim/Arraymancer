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

import strformat

import ../tensor/tensor,
        math

# Needed for the partition function
from private/p_logsumexp import logsumexp


type Idx = SomeInteger


proc compute_scores[T](
  result: var Tensor[T],  # (B, ) - not nil
  input: Tensor[T],       # (T, B, num_tags)
  mask: Tensor[T],        # (T, B)
  transitions: Tensor[T], # (num_tags + 2, num_tags + 2)
  tags: Tensor[Idx],      # (T, B)
  timesteps, batch_size, hidden_dim: int,
  bos_tag, eos_tag: Idx
) =
  ## Computes the un-normalized log probabilities (combination of emissions and
  ## transition scores at each timestep).
  ##
  ## Returns:
  ##  - A Tensor[T] of non-normalized emission scores of shape [batch_size, ]

  # DEBUG
  echo (timesteps, batch_size, hidden_dim)
  echo input.shape

  # Transitions from bos_tag -> tag at time = 0 for all batches
  var transition_scores = index_select(transitions[bos_tag, _], axis = 1,
                                      indices = tags[0, _].squeeze()).squeeze()

  when compileOption("boundChecks"):
    doAssert result.shape == [batch_size], "Result should be of shape" &
      fmt" {batch_size} but got {result.shape}"
    doAssert transition_scores.shape == [batch_size], "Transition scores" &
      fmt" should be of shape {batch_size} but got {transition_scores.shape}"

  # Emission scores for tag at t = 0 for all in batch
  # Unoptimized - simple loop
  var emission_scores = newTensorUninit[input.T](batch_size)

  for i in 0 ..< batch_size:
    emission_scores[i] = input[0, i, tags[0, i]]
  
  when compileOption("boundChecks"): 
    doAssert emission_scores.shape == [batch_size], "Emission scores should" &
      fmt" be of shape {batch_size} but got {emission_scores.shape}"

  emission_scores .*= mask[0, _].squeeze()

  result += transition_scores + emission_scores

  # TODO: Optimize?
  for i in 1 ..< timesteps - 1:
    let 
      old_tags = tags[i - 1, _].squeeze(1)
      new_tags = tags[i, _].squeeze(1)

      old_mask = mask[i, _].squeeze()
      new_mask = mask[i + 1, _].squeeze()

    # New emission scores are the emission at time i for batch j to tag [i, j]
    for j in 0 ..< batch_size:
      emission_scores[i] = input[i, j, tags[i, j]]

    # New transition scores
    # This is applying transtion from old -> new tag across batch
    # Unoptimized version:
    # for j in 0 .. batch_size:
    #   transition_scores[j] = transitions[old_tags[j], new_tags[j]]
    transition_scores.apply3_inline(old_tags, new_tags):
      transitions[y, z]

    result += (transition_scores .* new_mask) + (emission_scores .* old_mask)
  
  # TODO: Make sure that last transition handled correctly 
  
  # Assume that masked when == 0
  let last_time_inds = (mask.sum(axis=0).squeeze() .- 1).astype(int)
  var last_tags = newTensorUninit[tags.T](batch_size)

  for i in 0 ..< batch_size:
    last_tags[i] = tags[last_time_inds[i], i]

  # Set transition scores to last_real_tag -> EOS_TAG across batch
  transitions[_, eos_tag].squeeze().index_select(axis=0, indices=last_tags, result=transition_scores)
  
  result += transition_scores

proc compute_log_partition_function[T](): Tensor[T] =
  ## Compute the partition function by using the forward algorithm to avoid
  ## explicitly calculating probabilties for all possible sequence
  ## configurations.
  discard

proc crf_forward*[T: SomeFloat](
  result: var Tensor[T],
  input: Tensor[T],
  mask: Tensor[T],
  transitions: Tensor[T],
  tags: Tensor[Idx],
  timesteps, batch_size, hidden_dim: int,
  bos_tag, eos_tag: Idx
) =
  ## Computes the log likelihood of input given transitions (emissions) matrix.
  ## Loss should be *negative* log likelihood.
  result = zeros[T](batch_size)
  result.compute_scores(input, mask, transitions, tags, timesteps, batch_size,
                        hidden_dim, bos_tag, eos_tag)
