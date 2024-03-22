import std / times
import ../../src/arraymancer

# The goal is to test the speed of various sigmoid implementation
# Some are numericall stable for positive, negative or both value

# We create a random tensor with randomly positive and negative value
let a = randomTensor(1000, 1000, 100.0f) -. 50.0f

proc sigmoid1[T: SomeFloat](t: Tensor[T]): Tensor[T] {.noInit.}=
  # Instable for large negative
  result = t.map_inline():
    1.T / (1.T + exp(-x))

proc sigmoid2[T: SomeFloat](t: Tensor[T]): Tensor[T] {.noInit.}=
  # Instable for large positive
  result = t.map_inline():
    let tmp = exp(x)
    tmp / (1.T + tmp)

proc sigmoid3[T: SomeFloat](t: Tensor[T]): Tensor[T] {.noInit.}=
  # Stable but branching in a loop
  result = t.map_inline():
    if x >= 0:
      1.T / (1 + exp(-x))
    else:
      let z = exp(x)
      z / (1 + z)

proc sigmoid4*[T: SomeFloat](t: Tensor[T]): Tensor[T] {.noInit.}=
  # Stable but expensive tanh
  result = t.map_inline():
    0.5.T * (tanh(0.5.T * x) + 1.T)

proc sigmoid5*[T: SomeFloat](t: Tensor[T]): Tensor[T] {.noInit.}=
  # Stable and probably fastest
  result = t.map_inline():
    let clip_x = max(-500, -x)
    1.T / (1 + exp(clip_x))

## Warmup for ondemand CPU
for i in 0..<1000:
  discard a.sigmoid1

var start = epochTime()
for i in 0..<1000:
  discard a.sigmoid1
echo " Sigmoid1: 1 / (1 + exp(-x)) ", epochTime() - start


start = epochTime()
for i in 0..<1000:
  discard a.sigmoid2
echo " Sigmoid2: exp(x) / (1 + exp(x)) ", epochTime() - start

start = epochTime()
for i in 0..<1000:
  discard a.sigmoid3
echo " Sigmoid3: branching ", epochTime() - start

start = epochTime()
for i in 0..<1000:
  discard a.sigmoid4
echo " Sigmoid4: 0.5 * (tanh(0.5 * x) + 1) ", epochTime() - start

start = epochTime()
for i in 0..<1000:
  discard a.sigmoid5
echo " Sigmoid5: 1 / (1 + exp(max(-500,-x)) ", epochTime() - start


##### Before 2017-11-02: with closures
# Results with -d:release on i5-5257U (dual-core mobile 2.7GHz, turbo 3.1)
# Note: results were done with cpuTime (single threaded)
# Sigmoid1: 1 / (1 + exp(-x)) 8.265147999999998
# Sigmoid2: exp(x) / (1 + exp(x)) 7.757116
# Sigmoid3: branching 12.477108
# Sigmoid4: 0.5 * (tanh(0.5 * x) + 1) 11.162277
# Sigmoid5: 1 / (1 + exp(max(-500,-x)) 10.050294

##### After 2017-11-02: with inline templates
# Note: results were done with cpuTime (single threaded)
#  Sigmoid1: 1 / (1 + exp(-x)) 6.620896
#  Sigmoid2: exp(x) / (1 + exp(x)) 6.654728999999998
#  Sigmoid3: branching 11.397973
#  Sigmoid4: 0.5 * (tanh(0.5 * x) + 1) 10.114952
#  Sigmoid5: 1 / (1 + exp(max(-500,-x)) 8.569589999999998

##### Multithreaded with epochTime
# We might have false sharing issue
#  Sigmoid1: 1 / (1 + exp(-x)) 6.127279000000001
#  Sigmoid2: exp(x) / (1 + exp(x)) 6.073428999999999
#  Sigmoid3: branching 10.575061
#  Sigmoid4: 0.5 * (tanh(0.5 * x) + 1) 9.359058000000001
#  Sigmoid5: 1 / (1 + exp(max(-500,-x)) 7.859188000000003
