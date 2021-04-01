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

import  ../../tensor/higher_order_applymap,
        ../../tensor,
        ../../autograd,
        math

# ############################################################
#
#             SGD: Stochastic Gradient Descent
#
# ############################################################

type
  Sgd*[TT] = object
    ## Stochastic gradient descent without momentum.
    params*: seq[Variable[TT]]
    lr*: TT.T # Learning rate. T is the generic parameter of Tensor[T]

proc newSGD*[T](params: varargs[Variable[Tensor[T]]], learning_rate: T): SGD[Tensor[T]] {.deprecated: "Use the optimizer macro instead".}=
  SGD[Tensor[T]](params: @params, lr: learning_rate)

proc update*(self: Sgd) =
  ## Performs an optimization update.
  ##
  ## Parameters:
  ## - ``self`` A SGD optimizer to update.
  ##
  ## This proc will update the weights in the model associated with the input
  ## optimizer according to the following rule:
  ##    `w = w - lr * gradient`

  # Update the params with formula Value -= lr * gradient
  # Note: SGD expects gradient to be scaled by batchsize (done by default in Arraymancer)
  for v in self.params:
    # v.value -= learning rate * grad
    if v.requires_grad:
      forEach val in v.value,
              g in v.grad:
        val -= self.lr * g
      # Zero the gradient
      v.grad = v.value.zeros_like # TODO "setZero" instead of a new allocation

func optimizerSGD*[M, T](model: M, learning_rate: T): Sgd[Tensor[T]] =
  ## Create a SGD optimizer that will update the model weight
  ##
  ## Parameters:
  ## - ``model`` Model to optimize.
  ## - ``learning_rate`` Learning rate.
  ##
  ## Returns:
  ## - A SGD optimizer with the given learning rate.
  ##
  ## Future TODO:
  ##   Rename to ``optimize[M](model: M, OptimizerKind: typedesc[SGD], learning_rate: SomeFloat):``

  # TODO: rename to optimize[M](model: M, OptimizerKind: typedesc[SGD], learning_rate: SomeFloat): ...
  # Pending https://github.com/nim-lang/Nim/issues/7734 and https://github.com/nim-lang/Nim/issues/7733

  result.params = @[]
  result.lr = learning_rate

  for layer in fields(model):
    when layer is Variable:
      result.params.add layer
    else:
      for field in fields(layer): # TODO recursive for any nesting depth of Model
        when field is Variable:
          result.params.add field

# ############################################################
#
#    SGDMomentum: Stochastic Gradient Descent with Momentum
#
# ############################################################

type
  SgdMomentum*[TT] = object
    ## Stochastic gradient descent with momentum.
    ## Details on Nesterov momentum can be found in
    ## `Sutskever et. al. 2013<http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf>`_
    params*: seq[Variable[TT]]
    lr*: TT.T                ## Learning rate
    momentum*: TT.T          ## Value of the momentum
    moments: seq[TT]         ## Moments for momentum
    decay: TT.T              ## Learning rate decay
    nesterov: bool           ## Flag for Nesterov momentum

proc update*(self: var SgdMomentum) =
  ## Performs an optimization update.
  ##
  ## Parameters:
  ## - ``self`` A SGDMomentum optimizer to update.
  ##
  ## This proc will update the weights in the model associated with the input
  ## optimizer according to the following rule:
  ##    `w = w - lr * gradient + m * moment`
  ## If nesterov is set to true then the following rule is applied instead:
  ##    `w = w - lr * gradient + m * v`
  ##
  ##    `v = - lr * gradient + m * moment`
  ## Where in both cases the `moment` is the gradient change applied in the
  ## previous update step and `m` is the momentum.
  ##
  ## If `decay` is greater than 0, the learning rate will be modified each
  ## call according to the following:
  ##      `lr = lr * 1/(1 + decay)`

  # Decay of the learning rate.
  self.lr *= 1 / (self.decay + 1)
  # Keeps track of decay without having to keep track of iterations.
  # Each update call is counted as one iteration.
  self.decay += self.decay
  for i in 0 ..< self.params.len:
    let v = self.params[i]
    if v.requires_grad:
      # This implementation of both kinds of momentum follows that of Tensorflow
      # and closely mirrors the formulation of Sustkever et. al.
      # Update the moments with the previous update.
      forEach mi in self.moments[i],
              g in v.grad:
        mi = self.momentum * mi - self.lr * g

      # When momentum = 0 this acts identically to SGD without momentum.
      if self.nesterov:
        # Update the params with formula Value = value - lr * gradient + momentum * v
        # where v = - lr * gradient + momentum * moment
        forEach val in v.value,
                g in v.grad,
                mi in self.moments[i]:
          val += -(self.lr * g) + self.momentum * mi
      else:
        # Update the params with formula Value = value - lr * gradient + momentum * moment
        forEach val in v.value,
                mi in self.moments[i]:
          val += mi

      # Zero the gradient
      v.grad = v.value.zeros_like # TODO "setZero" instead of a new allocation

func optimizerSGDMomentum*[M, T](model: M, learning_rate: T, momentum = T(0.0),
                                 decay = T(0.0), nesterov = false): SgdMomentum[Tensor[T]] =
  ## Create a SGD optimizer with optional momentum that will update the model weight
  ##
  ## Parameters:
  ## - ``model`` Model to optimize.
  ## - ``learning_rate`` Learning rate.
  ## - ``momentum`` Momentum.
  ## - ``decay`` How much the learning rate will decay each update.
  ## - ``nesterov`` Whether to use Nesterov momentum or not.
  ##
  ## Returns:
  ## - A SGD optimizer with momentum with the given parameters.
  ##
  ## Future TODO:
  ##   Rename to ``optimize[M](model: M, OptimizerKind: typedesc[SGDMomentum], learning_rate: SomeFloat):``

  # TODO: rename to optimize[M](model: M, OptimizerKind: typedesc[SGD], learning_rate: SomeFloat): ...
  # Pending https://github.com/nim-lang/Nim/issues/7734 and https://github.com/nim-lang/Nim/issues/7733

  result.params = @[]
  result.lr = learning_rate
  result.momentum = momentum
  result.decay = decay
  result.nesterov = nesterov

  for layer in fields(model):
    when layer is Variable:
      result.params.add layer
      result.moments.add layer.grad.zeros_like
    else:
      for field in fields(layer): # TODO recursive for any nesting depth of Model
        when field is Variable:
          result.params.add field
          result.moments.add field.grad.zeros_like

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
    beta1_t, beta2_t: TT.T          ## Current decay
    first_moments: seq[TT]          ## Exponential moving averages (mean estimation)
    second_moments: seq[TT]         ## Exponential moving averages squared (uncentered variance)
    epsilon: TT.T                   ## Epsilon for numerical stability when dividing

proc update*(self: var Adam) =
  # We use the second formulation of Adam from Kingma et al similar to Tensorflow

  # Bias corrected learning rate
  let lr_t = self.learning_rate * sqrt(1 - self.beta2_t) / (1 - self.beta1_t)

  # Raise β1^t and β2^t for next update
  self.beta1_t *= self.beta1
  self.beta2_t *= self.beta2

  for i in 0 ..< self.params.len:
    let v = self.params[i]
    if v.requires_grad:
      # Update biaised first moment estimate
      forEach fmi in self.first_moments[i], g in v.grad:
        fmi = self.beta1 * fmi + (1 - self.beta1) * g
      # Update biaised second moment estimate
      forEach smi in self.second_moments[i], g in v.grad:
        smi = self.beta2 * smi + (1 - self.beta2) * g * g
      # Adjust weight
      forEach val in v.value,
              fmi in self.first_moments[i],
              smi in self.second_moments[i]:
        val -= lr_t * fmi / (smi.sqrt + self.epsilon)

      # Zero the gradient
      v.grad = v.value.zeros_like # TODO "setZero" instead of a new allocation

proc optimizerAdam*[M, T](
        model: M,
        learning_rate: T = T(0.001),
        beta1 = T(0.9), beta2 = T(0.999),
        eps = T(1e-8)
      ): Adam[Tensor[T]] =
  ## Create a Adam optimizer that will update the model weight

  # TODO: rename to optimize[M](model: M, OptimizerKind: typedesc[SGD], learning_rate: SomeFloat): ...
  # Pending https://github.com/nim-lang/Nim/issues/7734 and https://github.com/nim-lang/Nim/issues/7733

  result.params = @[]
  result.learning_rate = learning_rate
  result.beta1 = beta1
  result.beta1_t = beta1
  result.beta2 = beta2
  result.beta2_t = beta2
  result.epsilon = eps

  for layer in fields(model):
    when layer is Variable:
      result.params.add layer
      result.first_moments.add layer.grad.zeros_like
      result.second_moments.add layer.grad.zeros_like
    else:
      for field in fields(layer): # TODO recursive for any nesting depth of Model
        when field is Variable:
          result.params.add field
          result.first_moments.add field.grad.zeros_like
          result.second_moments.add field.grad.zeros_like

# ############################################################
#
#                 Generic optimizers
#
# ############################################################

type
  Optimizer*[TT] = Sgd[TT] or Adam[TT]

proc zeroGrads*(o: Optimizer) =
  # Reset the gradients of the optimized params
  # TODO setZero instead of allocating.
  for v in o.params:
    v.grad = v.value.zeros_like
