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
  ../../src/arraymancer, aux_rosenbrock, ../testutils,
  unittest, random, strformat

# ############################################################
#
#                    Test suite
#
# ############################################################

const
  # Keep these around so we know what these values are.
  a = 1'f64
  b = 100'f64
  num_epochs = 100

let start_x = [1.5'f64].toTensor
let start_y = [1.5'f64].toTensor
type Model = object
  x, y: Variable[Tensor[float64]]

suite "[Optimizer] Optimizer on the Rosenbrock function":
  setup:
    let ctx = newContext Tensor[float64]
    let model = Model(x: ctx.variable(start_x.clone, requires_grad = true),
                      y: ctx.variable(start_y.clone, requires_grad = true))

  test "Stochastic gradient descent - Momentum":
    var optim = optimizerSGDMomentum[model, float64](model, learning_rate = 1e-4, momentum = 0.95)

    for epoch in 1 .. num_epochs:
      var s = &"Epoch {epoch:>3}/{num_epochs} - "
      s &= &"Rosenbrock({model.x.value[0]:>.12f}, {model.y.value[0]:>.12f}) = "
      s &= &"{rosenbrock(model.x.value, model.y.value)[0]:>.12f}"
      echo s
      (model.x.grad, model.y.grad) = drosenbrock(model.x.value, model.y.value)
      optim.update()


  test "Stochastic gradient descent - Nesterov Momentum":
    var optim = optimizerSGDMomentum[model, float64](model, learning_rate = 1e-4, momentum = 0.95, nesterov=true
    )
    for epoch in 1 .. num_epochs:
      var s = &"Epoch {epoch:>3}/{num_epochs} - "
      s &= &"Rosenbrock({model.x.value[0]:>.12f}, {model.y.value[0]:>.12f}) = "
      s &= &"{rosenbrock(model.x.value, model.y.value)[0]:>.12f}"
      echo s
      (model.x.grad, model.y.grad) = drosenbrock(model.x.value, model.y.value)
      optim.update()
