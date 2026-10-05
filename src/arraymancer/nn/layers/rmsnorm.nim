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

type RMSNormGate*[TT] {.final.} = ref object of Gate[TT]
  input, weight: Variable[TT]
  eps: getSubType(TT)

proc rmsnorm_backward_ag[TT](self: Gate[TT], payload: Payload[TT]): SmallDiffs[TT] =
  let self = RMSNormGate[TT](self)
  result = newDiffs[TT](2)
  var gi, gw: TT
  rmsnorm_backward(payload.variable.grad, self.input.value, self.weight.value, gi, gw, self.eps)
  if self.input.requires_grad: result[0] = gi
  if self.weight.requires_grad: result[1] = gw

proc rmsnorm_cache[TT](result: Variable[TT], input, weight: Variable[TT], eps: getSubType(TT)) =
  var gate = RMSNormGate[TT](input: input, weight: weight, eps: eps)
  result.grad = zeros_like(result.value)
  result.requires_grad = true
  register_node("RMSNorm", gate, rmsnorm_backward_ag[TT], result, input, weight)

proc rms_norm*[TT](input, weight: Variable[TT], eps: getSubType(TT) = getSubType(TT)(1e-5)): Variable[TT] =
  when compileOption("boundChecks"):
    check_ctx(input, weight)
    if weight.value.shape != [input.value.shape[^1]].toMetadata:
      raise newException(ValueError, "RMSNorm weight shape must be [D] matching input last dimension")

  new result
  result.context = input.context
  result.value = rmsnorm(input.value, weight.value, eps)
  if input.is_grad_needed or weight.is_grad_needed:
    result.rmsnorm_cache(input, weight, eps)

type
  RMSNorm*[T] = object
    weight*: Variable[Tensor[T]]
    eps*: T

proc init*[T](
  ctx: Context[Tensor[T]],
  layerType: typedesc[RMSNorm[T]],
  normalized_shape: int,
  eps: T = 1e-5.T
): RMSNorm[T] =
  result.weight = ctx.variable(ones[T]([normalized_shape]), requiresGrad = true)
  result.eps = eps

proc forward*[T](self: RMSNorm[T], input: Variable[Tensor[T]]): Variable[Tensor[T]] =
  input.rms_norm(self.weight, self.eps)

func outShape*[T](self: RMSNorm[T]): seq[int] = @[self.weight.value.shape[0]]
func inShape*[T](self: RMSNorm[T]): seq[int] = @[self.weight.value.shape[0]]
