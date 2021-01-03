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
        ./private/p_logsumexp

proc softmax*[T](input: Tensor[T]): Tensor[T] {.noInit.} =
  ## For each sample in a tensor:
  ##   do an exponential normalization of each of its class features xi
  ##   ``exp(xi) / ∑i exp(xi)``
  ##
  ## Input:
  ##   - A tensor of shape [batch_size, number_of_classes]
  ## Output:
  ##   - A tensor of shape [batch_size, number_of_classes]

  let batch_size = input.shape[0]
  result = zeros_like(input)

  # TODO: the performance of nested parallel regions is highly suspect
  for i in 0||(batch_size-1):
    let (max, sumexp) = input[i, _].streaming_max_sumexp

    var res_slice = result[i, _]

    forEachSerial r in res_slice, inpi in input[i, _]:
      r = stable_softmax(inpi, max, sumexp)
