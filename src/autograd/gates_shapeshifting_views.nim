# Copyright 2017-2018 Mamy André-Ratsimbazafy & the Arraymancer contributors
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

import  typetraits,
        ../tensor/tensor,
        ./autograd_common

template `[]`*[TT](v: Variable[TT], args: varargs[untyped]): Variable[TT] =
  ## Slice the tensor contained by the dynamic graph Variable
  ## Input:
  ##   - a Variable
  ## Output:
  ##   - a sliced Variable

  # TODO - investigate https://github.com/mratsim/Arraymancer/issues/241
  # As https://github.com/mratsim/Arraymancer/commit/e609e998d663710281dbe161249a0139befa818c
  # which fixed https://github.com/mratsim/Arraymancer/issues/185 had to be rollbacked

  # Ensure that v is only called once even if it's a function with side-effects
  let z = v

  # TODO: backprop support
  var result: type(z)
  new result

  type S = type z.value[args]

  result.context = z.context
  result.requires_grad = z.requires_grad
  when S is AnyTensor:
    result.value = z.value[args]
    if result.requires_grad:
      result.grad = z.grad[args]
  else: # Not sure how to backprop that
    result.value = [z.value[args]].toTensor
    if result.requires_grad:
      result.grad = [z.grad[args]].toTensor

  result

  # TODO: tests for slicing correspondence

# #############################################

type ReshapeGate*[TT] {.final.} = ref object of Gate[TT]
  cached_input_shape: MetadataArray

proc reshape_backward_ag[TT](self: ReshapeGate[TT], payload: Payload[TT]): SmallDiffs[TT] =
  let gradient = payload.variable.grad
  result = newDiffs[TT](1)
  result[0] = gradient.reshape(self.cached_input_shape)

proc reshape_cache[TT](result: Variable[TT], a: Variable[TT]) =
  # Gate
  var gate: ReshapeGate[TT]
  new gate
  gate.cached_input_shape = a.value.shape

  # Result setup
  result.grad = zeros_like(result.value)
  result.requires_grad = true

  # Add to graph
  register_node(
    "Reshape",
    gate,
    reshape_backward_ag[TT],
    result,
    a
  )

proc reshapeImpl[TT](a: Variable[TT], shape: MetadataArray): Variable[TT] =
  # Resulting var
  new result
  result.context = a.context
  result.value = a.value.reshape(shape)

  # Caching for backprop
  if a.is_grad_needed:
    result.reshape_cache(a)

proc reshape*[TT](a: Variable[TT], shape: varargs[int]): Variable[TT] =
  ## Input:
  ##   - A variable
  ##   - A shape
  reshapeImpl(a, shape.toMetadataArray)

proc reshape*[TT](a: Variable[TT], shape: MetadataArray): Variable[TT] =
  ## Input:
  ##   - A variable
  ##   - A shape
  reshapeImpl(a, shape)

proc flatten*[TT](a: Variable[TT]): Variable[TT] =
  ## Input:
  ##   - A variable
  reshapeImpl(a, [a.value.shape[0], a.value.size div a.value.shape[0]].toMetadata)

# ############################################################
#
#                   Squeeze / Unsqueeze
#
# ############################################################

template squeezeUnsqueeze(GateName, forward_proc, backward_proc: untyped): untyped =

  type GateName[TT] {.final.} = ref object of Gate[TT]
    axis: int

  proc `forward_proc _ backward _ ag`[TT](self: GateName[TT], payload: Payload[TT]): SmallDiffs[TT] =
    result = newDiffs[TT](1)
    result[0] = payload.variable.grad.backward_proc(self.axis)

  proc `forward_proc _ cache`[TT](result: Variable[TT], x: Variable[TT], axis: Natural) =
    result.requires_grad = true
    result.grad = zeros_like result.value

    # Gate
    var gate: GateName[TT]
    new gate
    gate.axis = axis

    register_node(
      GateName.name,
      gate,
      `forward_proc _ backward _ ag`[TT],
      result,
      x
    )

  proc forward_proc*[TT](v: Variable[TT], axis: Natural): Variable[TT] =
    # Resulting var
    new result
    result.context = v.context
    result.value = forward_proc(v.value, axis)

    # Caching for backprop
    if v.requires_grad:
      result.`forward_proc _ cache`(v, axis)

squeezeUnsqueeze(SqueezeGate, squeeze, unsqueeze)
squeezeUnsqueeze(UnsqueezeGate, unsqueeze, squeeze)
