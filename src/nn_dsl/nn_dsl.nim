# Copyright (c) 2018 Mamy André-Ratsimbazafy and the Arraymancer contributors
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

import dsl_core, dsl_types

export
  network,
  TrainableLayer,
  Conv2DLayer,
  LinearLayer

proc flatten*(s: openarray[int]): int {.inline.}=
  assert s.len != 0
  result = 1
  for val in s:
    result *= val

when isMainModule:

  import ../tensor/tensor, ../nn/nn, ../autograd/autograd

  var ctx: Context[Tensor[float32]]

  network ctx, FooNet:
    layers:
      x:   Input([1, 28, 28])                            # Real shape [N, 1, 28, 28]
      cv1: Conv2D(x.out_shape, 20, 5, 5)                 # Output shape [N, 20, 24, 24] (kernel 5x5, padding 0, stride 1)
      mp1: MaxPool2D(cv1.out_shape, (2,2), (0,0), (2,2)) # Output shape [N, 20, 12, 12] (kernel 2X2, padding 0, stride 2)
      classifier:
        Linear(mp1.out_shape.flatten, 10)                # Output shape [N, 10]
    forward x:
      x.cv1.relu.mp1.flatten.classifier


  let a = init(ctx, FooNet)

  echo a
