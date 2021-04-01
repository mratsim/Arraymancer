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

import  ../../tensor,
        ../../ml,
        ../../autograd

type MSELoss*[TT] {.final.} = ref object of Gate[TT]
  target: TT
  cache: Variable[TT]

proc mse_backward_ag[TT](self: MSELoss[TT], payload: Payload[TT]): SmallDiffs[TT] =
  let gradient = payload.variable.grad
  # Gradient is a tensor of shape 1
  assert gradient.shape == [1]
  let grad = gradient.data[gradient.offset]

  let norm = grad * 2'f32 / gradient.size.float32 # TODO divide by total number of elements or by batch size? https://github.com/pytorch/pytorch/issues/3322
                                                  # See also Stanford course: http://theory.stanford.edu/~tim/s15/l/l15.pdf

  result = newDiffs[TT](1)
  forEach r0 in result[0],
          v in self.cache.value,
          t in self.target:
    r0 = norm * (v - t)

proc mse_cache[TT](result: Variable[TT], input: Variable[TT], target: TT) =
  ## We expect input with shape [batch_size, features]

  # Gate
  var gate: MSEloss[TT]
  new gate
  gate.cache = input
  gate.target = target

  # Result setup
  result.grad = zeros_like result.value
  result.requires_grad = true

  # Add to graph
  register_node(
    "Mean Squared Error",
    gate,
    mse_backward_ag[TT],
    result,
    input
  )

proc mse_loss*[TT](input: Variable[TT], target: TT): Variable[TT] =
  ## Mean square error loss function.
  ## Input:
  ##   - An input variable of predicted values of shape [batch_size, features]
  ##   - The ground truth of the same shape

  # Resulting var
  new result
  result.context = input.context
  # TODO: implement a Scalar[T] concept instead of rewrapping the result into a Tensor
  result.value = [mean_squared_error(input.value, target)].toTensor

  # Caching for backprop
  if input.is_grad_needed:
    result.mse_cache(input, target)
