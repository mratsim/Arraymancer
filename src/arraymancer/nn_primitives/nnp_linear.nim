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

import ../tensor

# Linear forward and backward

proc linear*[T](input, weight: Tensor[T], bias: Tensor[T], output: var Tensor[T]) {.inline.} =
  # Y = X Wᵀ + b, X: [..., in], W: [out, in], b: [1, out]
  let flatIn = input.reshape(input.size div weight.shape[1], weight.shape[1])
  var flatOut = flatIn * weight.transpose
  flatOut +.= bias

  var outShape = input.shape[0 ..< input.rank - 1]
  outShape.add weight.shape[0]
  output = flatOut.reshape(outShape)

proc linear*[T](input, weight: Tensor[T], output: var Tensor[T]) {.inline.} =
  # Y = X Wᵀ, X: [..., in], W: [out, in]
  let flatIn = input.reshape(input.size div weight.shape[1], weight.shape[1])

  var outShape = input.shape[0 ..< input.rank - 1]
  outShape.add weight.shape[0]
  output = (flatIn * weight.transpose).reshape(outShape)

proc linear_backward*[T](
        input, weight, gradOutput: Tensor[T],
        gradInput, gradWeight, gradBias: var Tensor[T]) {.inline.} =
  # dX = dY W, dW = dYᵀ X, db = Σ dY
  let
    flatIn = input.reshape(input.size div weight.shape[1], weight.shape[1])
    flatOut = gradOutput.reshape(gradOutput.size div weight.shape[0], weight.shape[0])
  gradInput = (flatOut * weight).reshape(input.shape)
  gradWeight = flatOut.transpose * flatIn
  gradBias = sum(flatOut, axis = 0)

proc linear_backward*[T](
        input, weight, gradOutput: Tensor[T],
        gradInput, gradWeight: var Tensor[T]) {.inline.} =
  # dX = dY W, dW = dYᵀ X
  let
    flatIn = input.reshape(input.size div weight.shape[1], weight.shape[1])
    flatOut = gradOutput.reshape(gradOutput.size div weight.shape[0], weight.shape[0])
  gradInput = (flatOut * weight).reshape(input.shape)
  gradWeight = flatOut.transpose * flatIn
