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

import  ../../private/ast_utils,
        ../../tensor/tensor,
        ../../ml/ml,
        ../../autograd/autograd,
        ./loss

type MSELoss*{.final.}[TT] = ref object of Loss[TT]
  cache: Variable[TT]
  # nb_grads, from Gate
  # target, from Loss

method forward*[TT](self: MSELoss[TT], input: Variable[TT], target: TT): Variable[TT] {.inline, locks:0.}=
  # We expect input with shape [batch_size, features]
  new result
  result.context = input.context

  # TODO: implement a Scalar[T] concept instead of rewrapping the result into a Tensor
  result.value = [mean_squared_error(input.value, target)].toTensor

method backward*[TT](self: MSELoss[TT], gradient: TT): SmallDiffs[TT] {.noInit, inline, locks:0.}=
  let norm = 2 / gradient.size
  result[0] = map2_inline(cache.value, gradient):
    norm * (x - y)

proc mean_squared_error*[TT](input: Variable[TT], target: Tensor[TT]): Variable[TT] =
  ## Mean square error loss function.
  ## Input:
  ##   - An input variable of predicted values of shape [batch_size, features]
  ##   - The ground truth of the same shape

  # Gate
  var gate: MSEloss[TT]
  new gate
  gate.nb_grads = 1

  # Node
  var node: Node[TT]
  new node

  node.gate = gate
  node.parents[0] = input.weakRef

  input.context.push(node)

  # Resulting var
  result = gate.forward(input, target)
  node.payload = result

  # Caching for backprop
  if input.is_grad_needed:
    result.grad = zeros_like(result.value)
    result.requires_grad = true

    gate.cache = input
    gate.target = target
