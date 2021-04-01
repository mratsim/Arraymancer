# Copyright 2017-Present Mamy André-Ratsimbazafy & the Arraymancer contributors
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

import
  ../../../src/arraymancer, ../../testutils,
  unittest, random, strformat

# ############################################################
#
#                    Auxiliary functions
#
# ############################################################

# We use the Rosenbrock function for testing optimizers
# https://en.wikipedia.org/wiki/Rosenbrock_function

proc rosenbrock[T](x, y: Tensor[T], a: static T = 1, b: static T = 100): Tensor[T] =
  # f(x, y) = (a - x)² + b(y - x²)²
  result = map2_inline(x, y):
    let u = a - x
    let v = y - x*x
    u * u + b * v * v

proc drosenbrock[T](x, y: Tensor[T], a: static T = 1, b: static T = 100): tuple[dx, dy: Tensor[T]] =
  result.dx = map2_inline(x, y):
    2.T*x - 2.T*a + 4*b*x*x*x - 4*b*x*y
  result.dy = map2_inline(x, y):
    2*b*(y - x*x)

# Sanity checks

block:
  let x = [1].toTensor
  let y = [1].toTensor
  doAssert rosenbrock(x, y) == [0].toTensor

block:
  let x = [5].toTensor
  let y = [25].toTensor
  doAssert rosenbrock(x, y, 5, 999) == [0].toTensor

block:
  let x = [1].toTensor
  let y = [1].toTensor
  doAssert drosenbrock(x, y) == ([0].toTensor, [0].toTensor)

# ############################################################
#
#                    Test suite
#
# ############################################################

testSuite "[Optimizer] Optimizer on the Rosenbrock function":
  const
    a = 1'f64
    b = 100'f64
    Batch =  1
    Epochs = 100

  let src_x = [1.5'f64].toTensor
  let src_y = [1.5'f64].toTensor

  # let src_x = randomTensor([Batch], 1.5'f64)
  # let src_y = randomTensor([Batch], 1.5'f64)
  let target_x = [a].toTensor.broadcast(Batch)
  let target_y = [a*a].toTensor.broadcast(Batch)

  type Model = object
    x, y: Variable[Tensor[float64]]

  test "Stochastic gradient descent":
    let ctx = newContext Tensor[float64]
    let model = Model(
      x: ctx.variable(src_x.clone, requires_grad = true),
      y: ctx.variable(src_y.clone, requires_grad = true)
    )

    let optim = optimizerSGD(model, learning_rate = 1e-4)

    echo "SGD - trying to optimize the Rosenbrock function"
    for epoch in 0 ..< Epochs:
      var s = &"Epoch {epoch:>3}/{Epochs} - Forward({model.x.value[0]:>.12f}, {model.y.value[0]:>.12f}) = "
      s &= &"{rosenbrock(model.x.value, model.y.value)[0]:>.12f}"
      echo s
      (model.x.grad, model.y.grad) = drosenbrock(model.x.value, model.y.value)
      optim.update()

  test "Adam (Adaptative Moment Estimation)":
    let ctx = newContext Tensor[float64]
    let model = Model(
      x: ctx.variable(src_x.clone, requires_grad = true),
      y: ctx.variable(src_y.clone, requires_grad = true)
    )

    var optim = optimizerAdam(model, learning_rate = 1e-4)

    echo "Adam - trying to optimize the Rosenbrock function"
    for epoch in 0 ..< Epochs:
      var s = &"Epoch {epoch:>3}/{Epochs} - Forward({model.x.value[0]:>.12f}, {model.y.value[0]:>.12f}) = "
      s &= &"{rosenbrock(model.x.value, model.y.value)[0]:>.4f}"
      echo s
      (model.x.grad, model.y.grad) = drosenbrock(model.x.value, model.y.value)
      optim.update()
