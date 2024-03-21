
import ../../src/arraymancer
import std / sequtils

let ctx = newContext Tensor[float32]

let
    a = ctx.variable(toSeq(1..12).toTensor.reshape(3,4).asType(float32))
    b = ctx.variable(toSeq(2..13).toTensor.reshape(3,4).asType(float32))
    c = ctx.variable(toSeq(3..11).toTensor.reshape(3,3).asType(float32))
    x = ctx.variable(toSeq(4..15).toTensor.reshape(4,3).asType(float32))
    y = ctx.variable(toSeq(5..16).toTensor.reshape(4,3).asType(float32))


# for t in [a,b,c,x,y]:
#   echo t.value


proc forwardNeuron[T](a,b,c,x,y: T): T =
  let
      ax = a * x
      by = b * y
      axpby = ax + by
      axpbypc = axpby + c
      # s = axpbypc.sigmoid()
  return axpbypc


var s = mean forwardNeuron(a,b,c,x,y)


echo s.value

s.backprop

echo a.grad

echo b.grad

echo c.grad

echo x.grad

echo y.grad
