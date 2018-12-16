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
        ../../autograd/autograd,
        ../../private/ast_utils

# ############################################################
#
#             SGD: Stochastic Gradient Descent
#
# ############################################################

# TODO: the following completely wrecks Nim
# with a worse case of https://github.com/mratsim/Arraymancer/issues/327

# type
#   Sgd*[T] = object
#     ## Stochastic gradient descent
#     params: seq[Variable[Tensor[T]]]
#     lr: T # Learning rate.

type
  Sgd*[TT] = object
    ## Stochastic gradient descent
    params*: seq[Variable[TT]]
    lr*: TT.T # Learning rate. T is the generic parameter of Tensor[T]

proc newSGD*[T](params: varargs[Variable[Tensor[T]]], learning_rate: T): SGD[Tensor[T]] {.deprecated: "Use the optimizer macro instead".}=
  SGD[Tensor[T]](params: @params, lr: learning_rate)

proc update*(self: Sgd) =
  # Update the params with formula Value -= lr * gradient
  # Note: SGD expects gradient to be scaled by batchsize (done by default in Arraymancer)
  for v in self.params:
    # v.value -= learning rate * grad
    if v.requires_grad:
      apply2_inline(v.value, v.grad):
        x - self.lr * y
      # Zero the gradient
      v.grad = v.value.zeros_like # TODO "setZero" instead of a new allocation

func optimizerSGD*[M, T](model: M, learning_rate: T): Sgd[Tensor[T]] =
  ## Create a SGD optimizer that will update the model weight

  # TODO: rename to optimize[M](model: M, OptimizerKind: typedesc[SGD], learning_rate: SomeFloat): ...
  # Pending https://github.com/nim-lang/Nim/issues/7734 and https://github.com/nim-lang/Nim/issues/7733

  result.params = @[]
  result.lr = learning_rate

  for layer in fields(model):
    for field in fields(layer): # TODO recursive for any nesting depth of Model
      if field is Variable:
        result.params.add field

# ############################################################
#
#             Adam: Adaptative Moment Estimation
#
# ############################################################

type
  Adam*[TT] = object
    ## Adaptative Moment Estimation
    params: seq[Variable[TT]]       ## Learnable weights
    learning_rate: TT.T
    beta1, beta2: TT.T              ## Decays on first and second moment
    first_moments: seq[TT]          ## Exponential moving averages (mean estimation)
    second_moments: seq[TT]         ## Exponential moving averages squared (uncentered variance)


# ############################################################
#
#                 Generic optimizers
#
# ############################################################

type
  Optimizer[TT] = Sgd[TT]

proc zeroGrads*(o: Optimizer) =
  # Reset the gradients of the optimized params
  # TODO setZero instead of allocating.
  for v in o.params:
    v.grad = v.value.zeros_like
