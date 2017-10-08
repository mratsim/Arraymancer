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

import  ../../tensor/tensor,
        ../../autograd/autograd,
        typetraits

type
  Optimizer*[T] = ref object {.inheritable.}
    # Base class for optimizer
    params*: seq[Variable[Tensor[T]]] # Todo: we can't specify a collection of generic types like AnyTensor currently
    lr*: T # Learning rate. Gradient update are scaled by the learning rate

method update*(self: Optimizer) {.base.} =
  # Forward for loss layers
  raise newException(ValueError, "update method is not implemented for " & $self.type.name)

proc zeroGrads*(o: Optimizer) =
  # Reset the gradients of the optimized params
  for v in o.params:
    v.grad = v.value.zeros_like

type SGD*{.final.}[T] = ref object of Optimizer[T]

proc newSGD*[T](params: varargs[Variable[Tensor[T]]], learning_rate: T): SGD[T] =
  SGD[T](params: @params, lr: learning_rate)

method update*(self: SGD) =
  # Update the params with formula Value -= lr * gradient
  # Note: SGD expects gradient to be scaled by batchsize (done by default in Arraymancer)
  for v in self.params:
    v.value -= self.lr * v.grad
    v.grad = v.value.zeros_like
