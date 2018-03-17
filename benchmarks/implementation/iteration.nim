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

proc getTime(): float =
  when not defined(openmp):
    cpuTime() # This is the most accurate
  else:
    epochTime() # cpuTime count the sum of time on each CPU when multithreading.

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

proc simple_iter[T](t:Tensor[T]): Tensor[T] {.noInit.}=
  result = newTensorUninit[T](t.shape)
  for val in result.mitems:
    val = 1.T

proc simple_clone[T](t:Tensor[T]): Tensor[T] {.noInit.}=
  result = newTensorUninit[T](t.shape)
  for dst, src in mzip(result, t):
    dst = src

var start = cpuTime()
warmup()
var stop = cpuTime()
echo "Warmup: " & $(stop - start) & "s"

proc main()=

  start = getTime()
  let a = ones[float](timeSteps, spaceSteps)
  stop = getTime()
  echo "Ones: " & $(stop - start) & " seconds for tensor of shape: " & $a.shape


  start = getTime()
  let b = a.clone1()
  stop = getTime()
  echo "Cloning impl 1 (enumerate): " & $(stop - start) & " seconds for tensor of shape: " & $b.shape

  start = getTime()
  let c = a.clone2()
  stop = getTime()
  echo "Cloning impl 2 (mzip): " & $(stop - start) & " seconds for tensor of shape: " & $c.shape

  start = getTime()
  let dc = a.clone_deepCopy()
  stop = getTime()
  echo "Cloning with deepCopy: " & $(stop - start) & " seconds for tensor of shape: " & $dc.shape

  start = getTime()
  let si = a.simple_iter()
  stop = getTime()
  echo "Simple iteration: " & $(stop - start) & " seconds for tensor of shape: " & $si.shape

  start = getTime()
  let sc = a.simple_clone()
  stop = getTime()
  echo "Simple clone: " & $(stop - start) & " seconds for tensor of shape: " & $sc.shape

  start = getTime()
  let d = b + c
  stop = getTime()
  echo "Summing: " & $(stop - start) & " seconds for tensor of shape: " & $d.shape

  start = getTime()
  let m = d.mean
  stop = getTime()
  echo "Reducing: " & $(stop - start) & " seconds for tensor of shape: " & $d.shape
  echo "Reduce mean result: " & $m

main()


##############
# On i5-5257U (Broadwell mobile dual core)

# Warmup: 0.426093s
# Ones: 2.92065 seconds for tensor of shape: [50000, 10000]
# Cloning impl 1 (enumerate): 8.120993 seconds for tensor of shape: [50000, 10000]
# Cloning impl 2 (mzip): 8.308195 seconds for tensor of shape: [50000, 10000]
# Cloning with deepCopy: 16.055578 seconds for tensor of shape: [50000, 10000]
# Simple iteration: 3.729559000000002 seconds for tensor of shape: [50000, 10000]
# Simple clone: 8.490787999999995 seconds for tensor of shape: [50000, 10000]
# Summing: 14.659691 seconds for tensor of shape: [50000, 10000]
# Reducing: 5.58793 seconds for tensor of shape: [50000, 10000]
# Reduce mean result: 2.0
# xtime.rb: 86.96s, 3767.0Mb

# OpenMP

# 369
# Warmup: 0.401529s
# Ones: 2.770177125930786 seconds for tensor of shape: [50000, 10000]
# Cloning impl 1 (enumerate): 5.454313993453979 seconds for tensor of shape: [50000, 10000]
# Cloning impl 2 (mzip): 5.626652002334595 seconds for tensor of shape: [50000, 10000]
# Cloning with deepCopy: 16.12829494476318 seconds for tensor of shape: [50000, 10000]
# Simple iteration: 4.012426137924194 seconds for tensor of shape: [50000, 10000]
# Simple clone: 9.845687866210938 seconds for tensor of shape: [50000, 10000]
# Summing: 13.06557416915894 seconds for tensor of shape: [50000, 10000]
# Reducing: 4.439347982406616 seconds for tensor of shape: [50000, 10000]
# Reduce mean result: 2.0
# 63.79s, 3886.6Mb
