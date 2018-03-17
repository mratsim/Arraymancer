import ../../src/arraymancer
import ../../src/tensor/backend/[openmp, memory_optimization_hints]
import times

const
  dz = 0.01
  z = 100
  spaceSteps = int(z / dz)
  timeSteps = 50000

proc warmup() =
  # Raise CPU to max perf even if using ondemand CPU governor
  # (mod is a costly operation)
  var foo = 123
  for i in 0 ..< 100000000:
    foo += i*i mod 456
    foo = foo mod 789
  echo foo

template map_implv1*[T](t: Tensor[T], op:untyped): untyped =

  let z = t # ensure that if t is the result of a function it is not called multiple times

  type outType = type((
    block:
      var x{.inject.}: type(items(z));
      op
  ))

  var dest = newTensorUninit[outType](z.shape)
  let data{.restrict.} = dest.dataArray # Warning ⚠: data pointed to will be mutated

  omp_parallel_blocks(block_offset, block_size, dest.size):
    for i, x {.inject.} in enumerate(z, block_offset, block_size):
      data[i] = op
  dest

template map_implv2*[T](t: Tensor[T], op:untyped): untyped =
  let z = t # ensure that if t is the result of a function it is not called multiple times

  type outType = type((
    block:
      var x{.inject.}: type(items(z));
      op
  ))

  var dest = newTensorUninit[outType](z.shape)
  omp_parallel_blocks(block_offset, block_size, dest.size):
    for d, x {.inject.} in mzip(dest, z, block_offset, block_size):
      d = op
  dest

proc clone1*[T](t: Tensor[T]): Tensor[T] {.noInit.}=
  result = t.map_implv1(x)

proc clone2*[T](t: Tensor[T]): Tensor[T] {.noInit.}=
  result = t.map_implv2(x)

proc clone_deepCopy[T](t:Tensor[T]): Tensor[T] {.noInit, inline.}=
  deepCopy(result, t)

var start = cpuTime()
warmup()
var stop = cpuTime()
echo "Warmup: " & $(stop - start) & "s"

proc main()=

  start = cpuTime()
  let a = ones[float](timeSteps, spaceSteps)
  stop = cpuTime()
  echo "Ones: " & $(stop - start) & " seconds for tensor of shape: " & $a.shape


  start = cpuTime()
  let b = a.clone1()
  stop = cpuTime()
  echo "Cloning impl 1 (enumerate): " & $(stop - start) & " seconds for tensor of shape: " & $b.shape

  start = cpuTime()
  let c = a.clone2()
  stop = cpuTime()
  echo "Cloning impl 2 (mzip): " & $(stop - start) & " seconds for tensor of shape: " & $c.shape

  start = cpuTime()
  let dc = a.clone_deepCopy()
  stop = cpuTime()
  echo "Cloning with deepCopy: " & $(stop - start) & " seconds for tensor of shape: " & $dc.shape

  start = cpuTime()
  let d = b + c
  stop = cpuTime()
  echo "Summing: " & $(stop - start) & " seconds for tensor of shape: " & $d.shape

  start = cpuTime()
  let m = d.mean
  stop = cpuTime()
  echo "Reducing: " & $(stop - start) & " seconds for tensor of shape: " & $d.shape
  echo "Reduce mean result: " & $m

main()


##############
# On i5-5257U (Broadwell mobile dual core)

# 369
# Warmup: 0.40981s
# Ones: 2.422835 seconds for tensor of shape: [50000, 10000]
# Cloning impl 1 (enumerate): 6.464892000000001 seconds for tensor of shape: [50000, 10000]
# Cloning impl 2 (mzip): 8.214280999999998 seconds for tensor of shape: [50000, 10000]
# Cloning with deepCopy: 15.701433 seconds for tensor of shape: [50000, 10000]
# Summing: 14.395039 seconds for tensor of shape: [50000, 10000]
# Reducing: 5.275849000000001 seconds for tensor of shape: [50000, 10000]
# Reduce mean result: 2.0
