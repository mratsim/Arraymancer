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

import  ../../tensor/[tensor, higher_order_applymap],
        ../../autograd/autograd

type
  Sgd*[TT] = object
    ## Stochastic gradient descent
    params*: seq[Variable[TT]]
    lr*: float32 # Learning rate.

  Optimizer[TT] = Sgd[TT]

proc zeroGrads*(o: Optimizer) =
  # Reset the gradients of the optimized params
  for v in o.params:
    v.grad = v.value.zeros_like

proc newSGD*[T](params: varargs[Variable[Tensor[T]]], learning_rate: T): SGD[Tensor[T]] {.deprecated: "Use the optimizer macro instead".}=
  SGD[Tensor[T]](params: @params, lr: learning_rate)

proc update*(self: Sgd) =
  # Update the params with formula Value -= lr * gradient
  # Note: SGD expects gradient to be scaled by batchsize (done by default in Arraymancer)
  for v in self.params:
    # v.value -= learning rate * grad
    apply2_inline(v.value, v.grad):
      x - self.lr * y
    v.grad = v.value.zeros_like

func optimizerSGD*[M](model: M, learning_rate: SomeReal): Sgd[Tensor[SomeReal]] =
  ## Create a SGD optimizer that will update the model weight

  # TODO: rename to optimize[M](model: M, OptimizerKind: typedesc[SGD], learning_rate: SomeReal): ...
  # Pending https://github.com/nim-lang/Nim/issues/7734 and https://github.com/nim-lang/Nim/issues/7733

  result.params = @[]
  result.lr = learning_rate

  for layer in fields(model):
  # when layer is TrainableLayer: # TODO: This is broken and generates two method declarations with the same name.
    result.params.add layer.weight
    result.params.add layer.bias
