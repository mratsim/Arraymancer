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

import  ../../tensor/tensor,
        math

# Implements numerically stable logsumexp

# - Classic version: log ∑i exp(xi) = α + log ∑i exp(xi−α)
# with α = max(xi) for xi in x

# - Streaming version: from http://www.nowozin.net/sebastian/blog/streaming-log-sum-exp-computation.html
# which is similar to Welford algorithm for streaming mean and variance in statistics

# Benchmark shows that the streaming version is 25% faster
# from batches of 256x3 classes to 256x1000 classes
# (see project_root/benchmark_implementation)

proc streaming_max_sumexp*[T](t: Tensor[T]): tuple[max:T, sumexp: T] {.noSideEffect, inline.}=
  result.max = -Inf.T   # will store the streaming max of the tensor
  result.sumexp = 0.T   # will store the streaming sum of exp of the tensor

  for x in t:
    if x <= result.max:
      result.sumexp += exp(x - result.max)
    else:
      result.sumexp *= exp(result.max - x)
      result.sumexp += 1
      result.max = x

proc logsumexp*[T: SomeReal](t: Tensor[T]): T =
  # Advantage:
  #  - Only one loop over the data
  #  - Can be done "on-the-fly"
  # Disadvantage:
  #  - no parallelization
  #  - branching in tight loop
  #
  # Note: most image problems have less than 1000 classes (or even 100)
  # However NLP problems may have 50000+ words in dictionary
  # It would be great to parallelize the one-pass version
  # (there are parallel running version algorithm we can draw inspiration from)

  # Also as problem size grow, the 1-pass version should scale much better
  # However so does parallel code. ==> Benchmark needed with low, medium and huge scale problem.

  let (max, sumexp) = t.streaming_max_sumexp
  result = max + ln(sumexp)


# proc logsumexp_classic[T: SomeReal](t: Tensor[T]): T =
#   # Advantage:
#   #  - OpenMP parallel
#   #  - No branching in a tight loop
#   # Disadvantage:
#   #  - Two loops over the data, might be costly if tensor is big
#   # Note: most image problems have less than 1000 classes (or even 100)
#   # However NLP problems may have 50000+ words in dictionary

#   let alpha = t.max

#   result = t.fold_inline() do:
#     # Init first element
#     x = exp(y - alpha)
#   do:
#     # Process next elements
#     x += exp(y - alpha)
#   do:
#     # Merge the partial folds
#     x += y

#   result = alpha + ln(result)