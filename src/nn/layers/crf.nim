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

import  ../../tensor/tensor,
        ../../nn_primitives/nn_primitives,
        ../../autograd/autograd


type Idx* = SomeInteger or SomeOrdinal

type CRFGate* [TT; Idx] {.final.} = ref object of Gate[TT]
  ## CRF (Linear) Gate for sequence prediction.
  transitions: Variable[TT]
  num_tags: Idx

  # Special values for 
  bos_tag: Idx
  eos_tag: Idx

proc crf_forward[TT, Idx](
  result: var Variable[TT],
  input: Variable[TT],
  mask: Variable[TT],
  transitions: Variable[TT],
  tags: Tensor[Idx],
  num_tags: int,
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

  #[crf_forward(
    input.value,

  )]#

proc crf_viterbi*[TT]() = discard

proc crf*[TT](
  input: Variable[TT],
  mask: Variable[TT],
  transitions: Variable[TT],
  tags: Option[Tensor[Idx]],
  num_tags: int,
  reduce: bool = false
): Variable[TT] =
  ## Input:
  ##   - An `x` Variable of shape [timesteps, batch_size, hidden_size]
  ##   - A `mask` Variable of shape [timesteps, batch_size] with is_grad_needed
  ##     set to 0.
  ##   - A `transitions` matrix of size (num_tags + 2, num_tags + 2)
  ##     The extra tags are for BOS / EOS tags.
  ##   - A `tags` tensor of shape [timesteps, batch_size, num_tags + 2] - only needed if
  ##     doing training.  If not training, then this can be nil.
  ## 
  ## Return:
  ##   - Negative log likelihood Tensor [batch_size, ]
  ##   - Logits for tag prediction of shape [batch_size, sequence_length, num_tags]
  when compileOption("boundChecks"):
    doAssert input.value.shape.len == 3, fmt"Expected input variable of rank 3, got shape of {input.value.shape}"
    doAssert mask.value.shape[0..1] == input.value.shape[0..1], fmt"Mask and input shapes do not match:" &
      fmt"got {mask.value.shape[0..2]} and {input.value.shape[0..2]}"
    doAssert transitions.value.shape == [num_tags + 2, num_tags + 2], "Expected transitions matrix shape to " &
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
      randomTensor[float32](timesteps, batch_size, hidden_dim, max=1.1),
      requires_grad = true
    )

    mask = ctx.variable(ones[float32](timesteps, batch_size))

    num_tags: int = 5

    transitions = ctx.variable(
      (randomTensor(num_tags + 2, num_tags + 2, max=2.0'f32) .- 1.0'f32),
      requires_grad = false
    )
  
  suite "Basic CRF tests":

    test "When pass in some(Tensor[int]) can call CRF":
      var tags = option(randomTensor(timesteps, batch_size, max=num_tags))
      let output = crf(input, mask, transitions, tags, num_tags)
      assert output.value.shape == [batch_size, ],
        fmt"Got output shape {output.value.shape}"
    
    test "When pass in none(Tensor[int]) get ValueError":
      expect ValueError:
        let output2 = crf(input, mask, transitions, none(Tensor[int]), num_tags)
