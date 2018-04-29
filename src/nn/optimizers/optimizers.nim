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
        typetraits

type
  Optimizer*[T] = object of RootObj
    # Base class for optimizer
    params*: seq[Variable[Tensor[T]]] # Todo: we can't specify a collection of generic types like AnyTensor currently
    lr*: T # Learning rate. Gradient update are scaled by the learning rate

  SGD*{.final.}[T] = object of Optimizer[T]

proc zeroGrads*[T](o: Optimizer[T]) =
  # Reset the gradients of the optimized params
  for v in o.params:
    v.grad = v.value.zeros_like

proc newSGD*[T](params: varargs[Variable[Tensor[T]]], learning_rate: T): SGD[T] {.deprecated: "Use the optimizer macro instead".}=
  SGD[T](params: @params, lr: learning_rate)

proc update*[T](self: SGD[T]) =
  # Update the params with formula Value -= lr * gradient
  # Note: SGD expects gradient to be scaled by batchsize (done by default in Arraymancer)
  for v in self.params:
    # v.value -= learning rate * grad
    apply2_inline(v.value, v.grad):
      x - self.lr * y
    v.grad = v.value.zeros_like
