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
        ../../nn_primitives/nn_primitives,
        ../../autograd/autograd,
        ./loss,
        typetraits

template gen_cross_entropy_loss(LossType, forward_proc, backward_proc: untyped) =
  # Template of softmax and sigmoid cross entropy losses

  type `LossType`* {.inject, final.} [TT] = ref object of Loss[TT]
    cache: Variable[TT]
    # nb_grads, from Gate
    # target, from Loss

  proc `forward_proc forward`[TT](self: LossType[TT], a: Variable[TT], target: TT): Variable[TT] {.inline.}=
    # We expect a in shape [batch_size, features]

    new result
    result.context = a.context

    # TODO: implement a Scalar[T] concept instead of rewrapping the result into a Tensor
    result.value = [forward_proc(a.value, target)].toTensor

  proc `forward_proc backward`[TT](self: LossType[TT], payload: Payload[TT]): SmallDiffs[TT] {.noInit.}=
    let gradient = payload.variable.grad
    result = newDiffs[TT](1)
    result[0] = backward_proc(gradient, self.cache.value, self.target)

  proc forward_proc*[TT](a: Variable[TT], target: TT): Variable[TT] =
    # Gate
    var gate: LossType[TT]
    new gate
    gate.cache = a
    gate.target = target

    # Resulting var
    result = gate.`forward_proc forward`(a, target)

    if a.is_grad_needed:
      result.grad = zeros_like result.value
      result.requires_grad = true

      register_node(
        LossType.name,
        gate,
        `forward_proc _ backward`[TT],
        result,
        a
      )

gen_cross_entropy_loss SigmoidCrossEntropyLoss, sigmoid_cross_entropy, sigmoid_cross_entropy_backward
gen_cross_entropy_loss SoftmaxCrossEntropyLoss, softmax_cross_entropy, softmax_cross_entropy_backward


type SparseSoftmaxCrossEntropyLoss*{.final.}[TT; Idx: SomeNumber or byte or char or enum] = ref object of SparseLoss[TT, Idx]
  cache: Variable[TT]
  # nb_grads, from Gate
  # target, from Loss

proc sparse_softmax_ce_forward[TT, Idx](self: SparseSoftmaxCrossEntropyLoss[TT, Idx], a: Variable[TT], target: Tensor[Idx]): Variable[TT] {.inline.}=
  # We expect a in shape [batch_size, features]
  new result
  result.context = a.context
  # TODO: implement a Scalar[T] concept instead of rewrapping the result into a Tensor
  result.value = [sparse_softmax_crossentropy(a.value, target)].toTensor

proc sparse_softmax_ce_backward[TT, Idx](self: SparseSoftmaxCrossEntropyLoss[TT, Idx], payload: Payload[TT]): SmallDiffs[TT] {.inline.}=
  let gradient = payload.variable.grad
  result = newDiffs[TT](1)
  result[0] = sparse_softmax_crossentropy_backward(gradient, self.cache.value, self.target)

proc sparse_softmax_crossentropy*[TT; Idx: SomeNumber or byte or char or enum](
        a: Variable[TT],
        target: Tensor[Idx]): Variable[TT] =
  # Gate
  var gate: SparseSoftmaxCrossEntropyLoss[TT, Idx]
  new gate

  # Resulting var
  result = gate.forward(a, target)

  # Caching for backprop
  if a.is_grad_needed:
    result.grad = zeros_like result.value
    result.requires_grad = true

    gate.cache = a
    gate.target = target

    register_node(
      "Sparse Softmax Cross-Entropy",
      gate,
      `forward_proc _ backward`[TT],
      result,
      a
    )
