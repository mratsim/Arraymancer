
import ../../src/arraymancer
import std / sequtils

let ctx = newContext Tensor[float32]

let
    a = ctx.variable(toSeq(1..12).toTensor.reshape(3,4).asType(float32), requires_grad = true)
    b = ctx.variable(toSeq(2..13).toTensor.reshape(3,4).asType(float32), requires_grad = true)
    c = ctx.variable(toSeq(3..14).toTensor.reshape(3,4).asType(float32), requires_grad = true)
    d = ctx.variable(toSeq(4..15).toTensor.reshape(3,4).asType(float32), requires_grad = true)

# for t in [a,b,c,x,y]:
#   echo t.value


proc foo[T](a,b,c,d: T): T =
  result = stack(a, a+b, c-d, axis=0).sum()


var s = foo(a, b, c, d)


echo s.value

s.backprop()

echo a.grad

echo b.grad

echo c.grad

echo d.grad
