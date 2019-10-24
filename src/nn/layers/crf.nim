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


import strformat
import options

import ../../tensor/tensor,
        ../../nn_primitives/nn_primitives,
        ../../nn/init,
        ../../autograd/autograd


type Idx* = SomeInteger

type CRFGate*[TT; Idx] {.final.} = ref object of Gate[TT]
  ## CRF (Linear) Gate for sequence prediction.
  transitions: Variable[TT]
  num_tags: Idx

  # Special values for state transitions
  bos_tag: Idx
  eos_tag: Idx

  dims: tuple[timesteps, batch_size, hidden_dim: Idx]


proc init_transitions_matrix*[T: SomeFloat](num_tags: Idx; range_val: T = T(
    0.1)): Tensor[T] =
  ## Create emissions matrix within bounds [-range, range], uniformly
  ## distributed.  The special transitions from [any, start] and [end, any] are
  ## set to be an arbitrarily low value to prevent prohibited transitions.
  ##
  ## Input:
  ##   The `num_tags` indicating how many real (non-special) tag values there are.
  ##   The `range_val` giving the scale to initialize transition values.
  ##
  ## Returns
  ##   The initialized transitions matrix of shape [num_tags + 2, num_tags + 2]

  # TODO: In future, allow for rules prohibiting / mandating certain transitions.
  let bos_tag, eos_tag = (num_tags, num_tags + 1)
  result = xavier_uniform(num_tags + 2, num_tags + 2, T) * range_val

  # Scale for a disallowed transition relative to the range value
  const disallowed_transition_scale = 100_000

  result[_, bos_tag] = disallowed_transition_scale * -1.0 * abs(range_val)
  result[eos_tag, _] = disallowed_transition_scale * -1.0 * abs(range_val)


proc crf_forward[TT, Idx](
  result: var Variable[TT];
  input: Variable[TT];
  mask: Variable[TT];
  transitions: Variable[TT];
  tags: Tensor[Idx];
  num_tags: int;
  reduce = false
) =
  ## Compute the negative log likelihood for each input sequence.
  ## If `reduce` is true, return
  var gate: CRFGate[TT, Idx]
  new gate

  gate.transitions = transitions
  gate.num_tags = num_tags

  gate.bos_tag = Idx(num_tags)
  gate.eos_tag = Idx(num_tags + 1)

  let
    timesteps = input.value.shape[0]
    batch_size = input.value.shape[1]
    hidden_dim = input.value.shape[2]

  gate.dims = (timesteps: timesteps, batch_size: batch_size,
               hidden_dim: hidden_dim)

  crf_forward(
    result.value,
    input.value,
    mask.value,
    transitions.value,
    tags,
    gate.dims.timesteps, gate.dims.batch_size, gate.dims.hidden_dim,
    gate.bos_tag, gate.eos_tag
  )

proc crf_viterbi*[TT]() = discard

proc crf*[TT](
  input: Variable[TT];
  mask: Variable[TT];
  transitions: Variable[TT];
  tags: Option[Tensor[Idx]];
  num_tags: int;
  reduce: bool = false
): Variable[TT] =
  ## Input:
  ##   - An `x` Variable of shape [timesteps, batch_size, num_tags]
  ##   - A `mask` Variable of shape [timesteps, batch_size] with is_grad_needed
  ##     set to 0.
  ##   - A `transitions` matrix of size (num_tags + 2, num_tags + 2)
  ##     The extra tags are for BOS / EOS tags.
  ##   - A `tags` tensor of shape [timesteps, batch_size] - only needed if
  ##     doing training.  If not training, then this can be nil.
  ##
  ## Returns:
  ##   - Negative log likelihood Tensor [batch_size, ]
  ##   - Logits for tag prediction of shape [batch_size, sequence_length, num_tags]
  when compileOption("boundChecks"):
    doAssert input.value.shape.len == 3, fmt"Expected input variable of rank 3" &
      ", got shape of {input.value.shape}"
    doAssert input.value.shape[2] == num_tags, fmt"Expected input variable to" &
      " emit {num_tags}, emitted {input.value.shape[2]}"
    doAssert mask.value.shape[0..1] == input.value.shape[0..1],
        fmt"Mask and input shapes do not match:" &
        fmt"got {mask.value.shape[0..2]} and {input.value.shape[0..2]}"
    doAssert transitions.value.shape == [num_tags + 2, num_tags + 2],
        "Expected transitions matrix shape to " &
        fmt"match ({num_tags+2}, {num_tags+2}), got {transitions.value.shape}"

  assert mask.requires_grad == false, "Mask should not need a gradient"

  new result
  result.context = input.context

  let doing_training = input.is_grad_needed() or transitions.is_grad_needed()

  if doing_training:
    if tags.isNone:
      raise newException(ValueError, "Tags must be non-nil when training")
  else:
    let tags_tensor = tags.get()
    result.crf_forward(input, mask, transitions, tags_tensor, num_tags)


when isMainModule:
  import unittest

  let ctx = newContext Tensor[float32]

  let (timesteps, batch_size, hidden_dim) = (8, 30, 10)

  let
    input = ctx.variable(
      randomTensor[float32](timesteps, batch_size, hidden_dim, max = 1.1),
      requires_grad = true
    )

    mask = ctx.variable(ones[float32](timesteps, batch_size))

    num_tags: int = 5

    transitions = ctx.variable(
      (randomTensor(num_tags + 2, num_tags + 2, max = 2.0'f32) .- 1.0'f32),
      requires_grad = false
    )

  suite "Basic CRF tests":

    test "When pass in some(Tensor[int]) can call CRF":
      var tags = option(randomTensor(timesteps, batch_size, max = num_tags))
      let output = crf(input, mask, transitions, tags, num_tags)
      assert output.value.shape == [batch_size, ],
        fmt"Got output shape {output.value.shape}"

    test "When pass in none(Tensor[int]) get ValueError":
      expect ValueError:
        let output2 = crf(input, mask, transitions, none(Tensor[int]), num_tags)
