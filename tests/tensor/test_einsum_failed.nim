import ../../src/arraymancer
import unittest, sequtils

# A list of tests that are supposed to fail and their error messages we
# expect. Set the `toRun` argument to true to see a test fail

proc main() =
  template toRun(run: static bool, actions: untyped): untyped =
    when run:
      block:
        actions

  toRun(false):
    let a = einsum(x, y):
      x[i] * y[i]
    discard "test_einsum_failed.nim(5, 18) Error: undeclared identifier: 'x'"

  toRun(false):
    let a = zeros[float](5)
    let b = einsum(a, 5):
      a[i] * a[i]
    discard "einsum.nim(53, 12) Error: Argument to `einsum` must be a number of defined tensors!"

  toRun(false):
    let a = zeros[float](5)
    let b = zeros[float](5)
    let c = einsum(a, b):
      c[i] = a[i, j] * b[j, j]
      a[i, j] * b[j, k]
    discard "einsum.nim(137, 12) `stmt.len == 1` There may only be a single statement in `einsum`!"

  toRun(false):
    let a = zeros[float](5)
    let b = zeros[int](5)
    let c = einsum(a, b):
      res[i,j] = a[i] * b[j]
    discard "einsum.nim(305, 14) Error: All tensors must be of the same type! a is of type float64 while b is of type int!"

main()
GC_fullCollect()
