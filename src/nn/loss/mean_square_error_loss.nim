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
        ../../ml/ml,
        ../../autograd/autograd,
        ./loss

type MSELoss*{.final.}[TT] = ref object of Loss[TT]
  cache: Variable[TT]
  # nb_grads, from Gate
  # target, from Loss

proc forward[TT](self: MSELoss[TT], input: Variable[TT], target: TT): Variable[TT] {.inline.}=
  # We expect input with shape [batch_size, features]
  new result
  result.context = input.context

  # TODO: implement a Scalar[T] concept instead of rewrapping the result into a Tensor
  result.value = [mean_squared_error(input.value, target)].toTensor

method backward*[TT](self: MSELoss[TT], payload: Payload[TT]): SmallDiffs[TT] {.noInit, inline.}=
  let gradient = payload.variable.grad
  # Gradient is a tensor of shape 1
  assert gradient.shape == [1]
  let grad = gradient.data[gradient.offset]

  let norm = grad * 2'f32 / gradient.size.float32 # TODO divide by total number of elements or by batch size? https://github.com/pytorch/pytorch/issues/3322
                                                  # See also Stanford course: http://theory.stanford.edu/~tim/s15/l/l15.pdf

  result = newDiffs[TT](1)
  result[0] = map2_inline(self.cache.value, self.target):
    norm * (x - y)

proc mse_loss*[TT](input: Variable[TT], target: TT): Variable[TT] =
  ## Mean square error loss function.
  ## Input:
  ##   - An input variable of predicted values of shape [batch_size, features]
  ##   - The ground truth of the same shape

  # Gate
  var gate: MSEloss[TT]
  new gate

  # Node
  var node: Node[TT]
  new node

  node.gate = gate
  node.parents = newParents[TT](1)
  node.parents[0] = input.weakRef

  input.context.push(node)

  # Resulting var
  result = gate.forward(input, target)
  node.payload = Payload[TT](kind: pkVar, variable: result)

  # Caching for backprop
  if input.is_grad_needed:
    result.grad = zeros_like result.value
    result.requires_grad = true

    gate.cache = input
    gate.target = target
