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
        ../../nn_primitives,
        ../../autograd,
        typetraits

template gen_cross_entropy_loss(LossType, forward_proc, backward_proc: untyped) =
  # Template of softmax and sigmoid cross entropy losses

  type `LossType`*[TT] {.inject, final.} = ref object of Gate[TT]
    target: TT
    cache: Variable[TT]

  proc `forward_proc _ backward _ ag`[TT](self: Gate[TT], payload: Payload[TT]): SmallDiffs[TT] =
    let self = LossType[TT](self)
    let gradient = payload.variable.grad
    result = newDiffs[TT](1)
    result[0] = backward_proc(gradient, self.cache.value, self.target)

  proc `forward_proc _ cache`[TT](result: Variable[TT], a: Variable[TT], target: TT) =
    ## We expect a in shape [batch_size, features]
    # Gate
    var gate: LossType[TT]
    new gate
    gate.cache = a
    gate.target = target

    # Result setup
    result.grad = zeros_like result.value
    result.requires_grad = true

    # add to graph
    register_node(
      LossType.name,
      gate,
      `forward_proc _ backward _ ag`[TT],
      result,
      a
    )

  proc forward_proc*[TT](a: Variable[TT], target: TT): Variable[TT] =
    # Resulting var
    new result
    result.context = a.context
    # TODO: implement a Scalar[T] concept instead of rewrapping the result into a Tensor
    result.value = [forward_proc(a.value, target)].toTensor

    # Caching for backprop
    if a.is_grad_needed:
      result.`forward_proc _ cache`(a, target)

gen_cross_entropy_loss SigmoidCrossEntropyLoss, sigmoid_cross_entropy, sigmoid_cross_entropy_backward
gen_cross_entropy_loss SoftmaxCrossEntropyLoss, softmax_cross_entropy, softmax_cross_entropy_backward

type SparseSoftmaxCrossEntropyLoss*[TT; Idx: SomeNumber or byte or char or enum] {.final.} = ref object of Gate[TT]
  target: Tensor[Idx]
  cache: Variable[TT]

proc sparse_softmax_ce_backward_ag[TT, Idx](self: Gate[TT], payload: Payload[TT]): SmallDiffs[TT] =
  let self = SparseSoftmaxCrossEntropyLoss[TT, Idx](self)
  let gradient = payload.variable.grad
  result = newDiffs[TT](1)
  result[0] = sparse_softmax_crossentropy_backward(gradient, self.cache.value, self.target)

proc sparse_softmax_ce_cache[TT, Idx](result: Variable[TT], a: Variable[TT], target: Tensor[Idx]) =
  # We expect a in shape [batch_size, features]
  result.grad = zeros_like result.value
  result.requires_grad = true

  # Gate
  var gate: SparseSoftmaxCrossEntropyLoss[TT, Idx]
  new gate

  gate.cache = a
  gate.target = target

  # Instantantiate with Idx but remove it from the signature
  # Why is it needed here but not for Embedding?
  let backward = cast[Backward[TT]](sparse_softmax_ce_backward_ag[TT, Idx])

  register_node(
    "Sparse Softmax Cross-Entropy",
    gate,
    backward,
    result,
    a
  )

proc sparse_softmax_crossentropy*[TT; Idx: SomeNumber or byte or char or enum](
        a: Variable[TT],
        target: Tensor[Idx]): Variable[TT] =
  # Resulting var
  new result
  result.context = a.context
  # TODO: implement a Scalar[T] concept instead of rewrapping the result into a Tensor
  result.value = [sparse_softmax_crossentropy(a.value, target)].toTensor

  # Caching for backprop
  if a.is_grad_needed:
    result.sparse_softmax_ce_cache(a, target)
