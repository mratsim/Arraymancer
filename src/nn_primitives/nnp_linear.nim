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
        math

# Linear forward and backward

proc linear*[T](input, weight: Tensor[T], bias: Tensor[T], output: var Tensor[T]) {.inline.} =
  # Linear (Dense) forward primitive with bias
  #   - input tensor shape [batch_size, in_features]
  #   - weight tensor shape [out_features, in_features]
  #   - bias tensor shape [batch_size, out_features]
  # Output does not need to be initialized to 0 or the proper shape, data will be overwritten
  # Output is: Y = x * W.transpose + b

  output = input * weight.transpose # TODO: with the transpose the non-matching rows and cols is confusing
  output .+= bias

proc linear*[T](input, weight: Tensor[T], output: var Tensor[T]) {.inline.} =
  # Linear (Dense) forward primitive with bias
  #   - input tensor shape [batch_size, in_features]
  #   - weight tensor shape [out_features, in_features]
  # Output does not need to be initialized to 0 or the proper shape, data will be overwritten
  # Output is: Y = x * W.transpose
  output = input * weight.transpose

proc linear_backward*[T](
        input,
        weight,
        bias,
        gradOutput: Tensor[T],
        gradInput,
        gradWeight,
        gradBias: var Tensor[T]) {.inline.} =
  # Linear (Dense) backward primitive with bias
  # Tensors are expected in a batch first shape [batch_size, n_features]
  # var Tensors do not need to be initialized to 0 or the proper shape, data will be overwritten
  gradInput = gradOutput * weight
  gradWeight = gradOutput.transpose * input

  gradBias = sum(gradOutput, axis=0) # https://mlxai.github.io/2017/01/10/a-modular-approach-to-implementing-fully-connected-neural-networks.html

proc linear_backward*[T](
        input,
        weight,
        gradOutput: Tensor[T],
        gradInput,
        gradWeight: var Tensor[T]) {.inline.} =
  # Linear (Dense) backward primitive without bias
  # Tensors are expected in a batch first shape [batch_size, n_features]
  gradInput = gradOutput * weight
  gradWeight = gradOutput.transpose * input

