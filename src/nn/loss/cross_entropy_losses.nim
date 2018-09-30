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

import  ../../private/sequninit,
        ../../tensor/tensor,
        ../../nn_primitives/nn_primitives,
        ../../autograd/autograd,
        ./loss

template gen_cross_entropy_loss(LossType, forward_proc, backward_proc: untyped) =
  # Template of softmax and sigmoid cross entropy losses

  type `LossType`* {.inject, final.} [TT] = ref object of Loss[TT]
    cache: Variable[TT]
    # nb_grads, from Gate
    # target, from Loss

  proc forward[TT](self: LossType[TT], a: Variable[TT], target: TT): Variable[TT] {.inline.}=
    # We expect a in shape [batch_size, features]

    new result
    result.context = a.context

    # TODO: implement a Scalar[T] concept instead of rewrapping the result into a Tensor
    result.value = [forward_proc(a.value, target)].toTensor

  method backward*[TT](self: LossType[TT], payload: Payload[TT]): SmallDiffs[TT] {.noInit, inline.}=
    let gradient = payload.variable.grad
    result = newSeqUninit[TT](1)
    result[0] = backward_proc(gradient, self.cache.value, self.target)

  proc forward_proc*[TT](a: Variable[TT], target: TT): Variable[TT] =
    # Gate
    var gate: LossType[TT]
    new gate
    gate.cache = a
    gate.target = target

    # Node
    var node: Node[TT]
    new node

    node.gate = gate
    node.parents = newSeqUninit[VariablePtr[TT]](1)
    node.parents[0] = a.weakRef

    a.context.push(node)

    # Resulting var
    result = gate.forward(a, target)
    node.payload = Payload[TT](kind: pkVar, variable: result)

    if a.is_grad_needed:
      result.grad = zeros_like result.value
      result.requires_grad = true

gen_cross_entropy_loss SigmoidCrossEntropyLoss, sigmoid_cross_entropy, sigmoid_cross_entropy_backward
gen_cross_entropy_loss SoftmaxCrossEntropyLoss, softmax_cross_entropy, softmax_cross_entropy_backward


type SparseSoftmaxCrossEntropyLoss* {.final.} [TT] = ref object of SparseLoss[TT]
  cache: Variable[TT]
  # nb_grads, from Gate
  # target, from Loss

proc forward[TT](self: SparseSoftmaxCrossEntropyLoss[TT], a: Variable[TT], target: Tensor[int]): Variable[TT] {.inline.}=
  # We expect a in shape [batch_size, features]

  new result
  result.context = a.context

  # TODO: implement a Scalar[T] concept instead of rewrapping the result into a Tensor
  result.value = [sparse_softmax_crossentropy(a.value, target)].toTensor

method backward*[TT](self: SparseSoftmaxCrossEntropyLoss[TT], payload: Payload[TT]): SmallDiffs[TT] {.noInit, inline.}=
  let gradient = payload.variable.grad
  result = newSeqUninit[TT](1)
  result[0] = sparse_softmax_crossentropy_backward(gradient, self.cache.value, self.target)

proc sparse_softmax_crossentropy*[TT](a: Variable[TT], target: Tensor[int]): Variable[TT] =
  # Gate
  var gate: SparseSoftmaxCrossEntropyLoss[TT]
  new gate

  # Node
  var node: Node[TT]
  new node

  node.gate = gate
  node.parents = newSeqUninit[VariablePtr[TT]](1)
  node.parents[0] = a.weakRef

  a.context.push(node)

  # Resulting var
  result = gate.forward(a, target)
  node.payload = Payload[TT](kind: pkVar, variable: result)

  # Caching for backprop
  if a.is_grad_needed:
    result.grad = zeros_like result.value
    result.requires_grad = true

    gate.cache = a
    gate.target = target
