import ../../src/arraymancer
import std / unittest

suite "Regression test - issue 459":
  test "Issue 459: tanh ambiguous identifier in NN DSL":
    # This test did not compile due to an ambiguous identifier error
    let (D_in, D_out) = (5, 1)

    network TestNet:
      layers:
        fc1: Linear(D_in, D_out)
      forward x:
        x.fc1.tanh

    var ctx = newContext Tensor[float32]
    let model {.used.} = ctx.init(TestNet)
