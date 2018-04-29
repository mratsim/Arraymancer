# Copyright (c) 2018 Mamy André-Ratsimbazafy and the Arraymancer contributors
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

import
  macros,
  dsl_core, dsl_types,
  ../nn/nn

export
  network,
  TrainableLayer,
  Conv2DLayer,
  LinearLayer

proc flatten*(s: openarray[int]): int {.inline.}=
  ## Flatten a tensor shape (i.e. returns the product)
  ## A tensor of shape [1, 2, 3] will have a shape [1*2*3]
  ## when flattened
  # TODO: make that work only at compile-time on a custom TopoShape type
  #       to avoid conflicts with other libraries.
  assert s.len != 0
  result = 1
  for val in s:
    result *= val

func optimizer*[M; U: SomeReal](model: M, optimizer: typedesc[Optimizer[U]], learning_rate: U): Optimizer[U] =
  result.params = @[]
  result.lr = learning_rate

  for layer in fields(model):
    when layer is TrainableLayer:
      result.params.add layer.weight
      result.params.add layer.bias

