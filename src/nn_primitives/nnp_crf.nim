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

import ../tensor/tensor,
        math

# Needed for the partition function
from private/p_logsumexp import logsumexp


type Idx = SomeInteger


proc compute_scores[T](
  input: Tensor[T],
  mask: Tensor[T],
  transitions: Tensor[T],
  tags: Tensor[Idx]
): Tensor[T] =
  ## Computes the un-normalized log probabilities (combination of emissions and
  ## transition scores at each timestep).
  ##
  ## Returns:
  ##  - A Tensor[T] of non-normalized emission scores of shape [batch_size, ]
  discard


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
  reduce: bool
) =
  ## Computes the log likelihood of input given transitions (emissions) matrix.
  ## Loss should be *negative* log likelihood.
  discard
