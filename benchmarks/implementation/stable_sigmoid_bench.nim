import times, ../../src/arraymancer, math

# The goal is to test the speed of various sigmoid implementation
# Some are numericall stable for positive, negative or both value

# We create a random tensor with randomly positive and negative value
let a = randomTensor(1000, 1000, 100.0f) .- 50.0f

proc sigmoid1[T: SomeReal](t: Tensor[T]): Tensor[T] =
  # Instable for negative
  proc sigmoid1_closure(x: T): T = 1.T / (1 + exp(-x))
  return t.map(sigmoid1_closure)

proc sigmoid2[T: SomeReal](t: Tensor[T]): Tensor[T] =
  # Instable for positive
  proc sigmoid2_closure(x: T): T =
    let z = exp(x)
    return z / (1.T + z)
  return t.map(sigmoid2_closure)

proc sigmoid3[T: SomeReal](t: Tensor[T]): Tensor[T] =
  # Stable but branching in a loop
  proc sigmoid3_closure(x: T): T =
    if x >= 0:
      return 1.T / (1 + exp(-x))
    let z = exp(x)
    return z / (1 + z)
  return t.map(sigmoid3_closure)

proc sigmoid4*[T: SomeReal](t: Tensor[T]): Tensor[T] =
  # Stable but expensive tanh
  proc sigmoid4_closure(x: T): T = 0.5.T * (tanh(0.5.T * x) + 1.T)
  return t.map(sigmoid4_closure)

proc sigmoid5*[T: SomeReal](t: Tensor[T]): Tensor[T] =
  # Stable and probably fastest
  proc sigmoid5_closure(x: T): T =
    let clip_x = max(-500, -x)
    return 1.T / (1 + exp(clip_x))
  return t.map(sigmoid5_closure)

## Warmup for ondemand CPU
for i in 0..<1000:
  discard a.sigmoid1

var start = cpuTime()
for i in 0..<1000:
  discard a.sigmoid1
echo " Sigmoid1: 1 / (1 + exp(-x)) ", cpuTime() - start


start = cpuTime()
for i in 0..<1000:
  discard a.sigmoid2
echo " Sigmoid2: exp(x) / (1 + exp(x)) ", cpuTime() - start

start = cpuTime()
for i in 0..<1000:
  discard a.sigmoid3
echo " Sigmoid3: branching ", cpuTime() - start

start = cpuTime()
for i in 0..<1000:
  discard a.sigmoid4
echo " Sigmoid4: 0.5 * (tanh(0.5 * x) + 1) ", cpuTime() - start

start = cpuTime()
for i in 0..<1000:
  discard a.sigmoid5
echo " Sigmoid5: 1 / (1 + exp(max(-500,-x)) ", cpuTime() - start


# Results with -d:release on i5-5257U (dual-core mobile 2.7GHz, turbo 3.1)
# Note: results vary strongly depending on your number of cores due to cpuTime methodology
# Sigmoid1: 1 / (1 + exp(-x)) 8.265147999999998
# Sigmoid2: exp(x) / (1 + exp(x)) 7.757116
# Sigmoid3: branching 12.477108
# Sigmoid4: 0.5 * (tanh(0.5 * x) + 1) 11.162277
# Sigmoid5: 1 / (1 + exp(max(-500,-x)) 10.050294