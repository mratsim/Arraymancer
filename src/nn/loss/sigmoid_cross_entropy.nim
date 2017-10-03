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

import ../../arraymancer_ag, ../../arraymancer, ../../autograd/utils
import ./loss
import math

proc check_input_target[T](input, target: Tensor[T]) {.inline.}=
  if input.shape != target.shape:
    raise newException(ValueError, "Input shape " & $input.shape &
      " and target shape " & $target.shape & " should be the same")

proc sigmoid_cross_entropy*[T](input, target: Tensor[T]): T {.inline.} =
  ## Sigmoid function + Cross-Entropy loss fused in one layer.
  ## This leverage the log-sum-exp trick for improved numerical stability
  ## It is also faster than calling both separately

  # Explanation

  # Cross-entropy has the following form for a single sample
  # CEi(yi, yi') = − ( ti ln(yi) + (1−ti) ln(1−yi) )

  # Since we pass a minibatch of several samples we should average by minibatch size (1/batchsize)
  # to keep the gradient magnitude/weight updates on the same scale as a single sample pass
  # CE(y, y') = − 1/n ∑i( ti ln(yi) + (1−ti) ln(1−yi) )

  # yi = ln(sigmoid(xi)) = ln(1/(1+e^-xi)) = ln(e^xi/( 1 + e^xi ))
  # yi = x - ln(1 + e^xi)

  # 1 - yi = ln(1 - sigmoid(xi)) = ln(1 + e^xi - e^xi) / (1 + e^xi))
  # 1 - yi = - ln(1 + e^xi)

  # Replacing Sigmoid Cross Entropy
  # SCE(x, y') = − 1/n ∑i(ti * (xi - ln(1 + e^xi)) + (1−ti) * -ln(1 + e^xi) )
  #            = − 1/n ∑i(ti * xi - ti * ln(1 + e^xi) -ln(1 + e^xi) + ti * ln(1 + e^xi) )
  #            = − 1/n ∑i(ti * xi - ln(1 + e^xi) )
  #            = − 1/n ∑i(ti * xi - ln(e^0 + e^xi) )
  #
  # Using the logsumexp trick with factorize by a constant
  # c = max(xi, 0)
  #
  # SCE(x, y') = − 1/n ∑i(ti * xi - ln(e^c *( e^(0-c) + e^(xi-c))
  #            = − 1/n ∑i(ti * xi - ln(e^c *( e^(0-c) + e^(xi-c))
  #            = − 1/n ∑i(ti * xi - c - ln(e^-c + e^(xi-c))
  #
  # If c = xi (xi > 0), ln(e^-c + e^(xi-c)) becomes ln(e^-xi + 1)
  # else c = 0 (xi < 0 ), ln(e^-c + e^(xi-c)) becomes ln(1 + e^xi)
  # Both cases are covered by ln(1 + e^-|xi|)
  #
  # Finally
  # SCE(x, y') = − 1/n ∑i(ti * xi - max(xi,0) - ln(1 + e^-|xi|)
  #
  #
  #
  # Other idea: streaming maximum (http://www.nowozin.net/sebastian/blog/streaming-log-sum-exp-computation.html)
  #

  # TODO: term rewriting macro for auto fusion

  when compileOption("boundChecks"):
    check_input_target(input, target)

  result = 0.T
  for xi, ti in zip(input, target):
    result += (-ti * xi +  max(xi,0) + ln(1 + exp(-abs(xi))) ) / T(input.shape[1]) #input.shape[1] is the batch size


type SigmoidCrossEntropyLoss* {.final.} [TT] = ref object of Loss[TT]
  cache: Variable[TT]

method forward*[TT](self: SigmoidCrossEntropyLoss[TT], a: Variable[TT], target: TT): Variable[TT] {.inline, locks:0.}=
  new result

  result.tape = a.tape
  # We expect a in shape @[features, batch_size]
  result.value = [a.value.sigmoid_cross_entropy(target)].toTensor

  result.grad = zeros[getSubType(TT)](1)


method backward*[TT](self: SigmoidCrossEntropyLoss[TT], gradient: TT): SmallDiffs[TT] {.inline, locks:0.}=

  # Derivative of Sigmoid-CE:
  # We start from this formula: SCE(x, y') = − 1/n ∑i(ti * xi - ln(1 + e^xi) )
  #                                        = 1/n ∑i(-ti * xi + ln(1 + e^xi) )
  #
  # On a single sample:
  # dSCE/dxi = d/dxi (-ti * xi + ln(1 + e^xi))
  #          = -ti + e^xi * 1/(1 + e^xi))
  #          = -ti * sigmoid(xi)
  #
  # For a vector of samples
  # dSCE/dx = 1/n ∑i( sigmoid(xi) - ti )

  # Gradient is a tensor of rank 1 with a single scalar value that we can access with gradient.data[gradient.offset]
  # The batch size is in self.cache.shape[0]
  # We can't directly use those in methods' closure for ome reason
  let previous_grad = gradient.data[gradient.offset]
  let batch_size = self.cache.value.shape[0]

  proc sigmoid_cross_entropy_backward_closure[T](xi, ti: T): T =
    previous_grad * ( 1.T / (1.T + exp(-xi)) - ti) / T(batch_size)

  result[0] = map2(self.cache.value, sigmoid_cross_entropy_backward_closure, self.target)

proc sigmoid_cross_entropy*[TT](a: Variable[TT], target: TT): Variable[TT] =

  # Gate
  var gate: SigmoidCrossEntropyLoss[TT]
  new gate
  gate.arity = 1
  gate.cache = a
  gate.target = target

  # Node
  var node: Node[TT]
  new node

  node.gate = gate
  node.parents[0] = a

  a.tape.push(node)

  # Resulting var
  result = gate.forward(a, target)
  result.ancestor = node
  node.child = result