# Copyright 2026 the Arraymancer contributors
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
        ../../private/ast_utils

type LayerNormGate*[TT] {.final.} = ref object of Gate[TT]
  input, weight, bias: Variable[TT]
  eps: getSubType(TT)

proc layernorm_backward_ag[TT](self: Gate[TT], payload: Payload[TT]): SmallDiffs[TT] =
  let self = LayerNormGate[TT](self)
  let gradOutput = payload.variable.grad
  var gi, gw: TT
  if self.bias.isNil:
    result = newDiffs[TT](2)
    layernorm_backward(gradOutput, self.input.value, self.weight.value, gi, gw, self.eps)
    if self.input.requires_grad: result[0] = gi
    if self.weight.requires_grad: result[1] = gw
  else:
    var gb: TT
    result = newDiffs[TT](3)
    layernorm_backward(gradOutput, self.input.value, self.weight.value, gi, gw, gb, self.eps)
    if self.input.requires_grad: result[0] = gi
    if self.weight.requires_grad: result[1] = gw
    if self.bias.requires_grad: result[2] = gb

proc layernorm_cache[TT](result: Variable[TT], input, weight, bias: Variable[TT], eps: getSubType(TT)) =
  var gate = LayerNormGate[TT](input: input, weight: weight, bias: bias, eps: eps)
  result.grad = zeros_like(result.value)
  result.requires_grad = true
  if bias.isNil:
    register_node("LayerNorm", gate, layernorm_backward_ag[TT], result, input, weight)
  else:
    register_node("LayerNorm", gate, layernorm_backward_ag[TT], result, input, weight, bias)

proc layer_norm*[TT](
    input, weight: Variable[TT],
    bias: Variable[TT] = nil,
    eps: getSubType(TT) = getSubType(TT)(1e-5)
  ): Variable[TT] =
  when compileOption("boundChecks"):
    check_ctx(input, weight)
    if weight.value.shape != [input.value.shape[^1]].toMetadata:
      raise newException(ValueError, "LayerNorm weight shape must be [D] matching input last dimension")
    if not bias.isNil:
      check_ctx(input, bias)
      if bias.value.shape != [input.value.shape[^1]].toMetadata:
        raise newException(ValueError, "LayerNorm bias shape must be [D] matching input last dimension")

  new result
  result.context = input.context
  result.value = if bias.isNil:
    layernorm(input.value, weight.value, eps = eps)
  else:
    layernorm(input.value, weight.value, bias.value, eps = eps)
  if input.is_grad_needed or weight.is_grad_needed or (not bias.isNil and bias.is_grad_needed):
    result.layernorm_cache(input, weight, bias, eps)

type
  LayerNorm*[T] = object
    weight*: Variable[Tensor[T]]
    bias*: Variable[Tensor[T]]
    eps*: T

proc init*[T](
  ctx: Context[Tensor[T]],
  layerType: typedesc[LayerNorm[T]],
  normalized_shape: int,
  bias = false,
  eps: T = 1e-5.T
): LayerNorm[T] =
  result.weight = ctx.variable(ones[T]([normalized_shape]), requiresGrad = true)
  if bias:
    result.bias = ctx.variable(zeros[T]([normalized_shape]), requiresGrad = true)
  result.eps = eps

proc forward*[T](self: LayerNorm[T], input: Variable[Tensor[T]]): Variable[Tensor[T]] =
  input.layer_norm(self.weight, self.bias, self.eps)

func outShape*[T](self: LayerNorm[T]): seq[int] = @[self.weight.value.shape[0]]
func inShape*[T](self: LayerNorm[T]): seq[int] = @[self.weight.value.shape[0]]
