import arraymancer
import unittest

suite "Regression test - issue 459":
  test "Issue 459: tanh ambiguous identifier in NN DSL":
    # This test did not compile due to an ambiguous identifier error
    let (D_in, D_out) = (5, 1)
    var ctx = newContext Tensor[float32]

    network ctx, TestNet:
      layers:
        fc1: Linear(D_in, D_out)
      forward x:
        x.fc1.tanh

    let model = ctx.init(TestNet)
