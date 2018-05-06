import math, times, strformat
import ../src/arraymancer

proc getTime(): float =
  when not defined(openmp):
    cpuTime() # This is the most accurate
  else:
    epochTime() # cpuTime count the sum of time on each CPU when multithreading.

const
  dz = 0.1
  z = 1000
  spaceSteps = int(z / dz)
  timeSteps = 50_000
  totalTime = 1_000_000
  dt = totalTime / timeSteps
  alpha = 2e-4
  startingTemp = 30.0
  oscillations = 20.0
  a_dz2 = alpha / dz^2


proc f(T: Tensor[float]): Tensor[float] =
  a_dz2 * (T[_, 0..^3] - 2.0 * T[_, 1..^2] + T[_, 2..^1])

proc eulerSolve(Ts: var Tensor[float]) =
  for t in 0 ..< timeSteps-1:
    Ts[t+1, 1..^2] = Ts[t, 1..^2] + dt * f(Ts[t, _])
    Ts[t+1, ^1] = Ts[t+1, ^2]

proc main() =
  var Ts = newTensorWith[float]([timeSteps, spaceSteps], startingTemp)


  for j in 0 ..< timeSteps:
    Ts[j, 0] = startingTemp - oscillations * sin(2.0 * PI * j.float / timeSteps)

  Ts.eulerSolve()
  echo Ts[45_000, 10]
  echo Ts[45_000, 100]
  echo Ts[45_000, 500]


let start = getTime()
main()
let stop = getTime()

let elapsed = stop - start
echo &"Arraymancer Euler solve - time taken: {elapsed} seconds"

#############################################################
# Single-threaded

# Measurement on i5-5257U (Dual core mobile Broadwell 2.7Ghz)
# Arraymancer 0.4.0 and Nim devel after Nim perf regression fix 88cf6573e04bd7ee8762aa336460b9748f0d4644

# Arraymancer Euler solve - time taken: 6.198931 seconds
# Measured by xtime.rb: 6.48s, 3854.8Mb

####################

# Measurement on i7-970 (Hexa core 3.2GHz) (Note: measurement before the perf regression fix)
# Arraymancer Euler solve - time taken: 5.060707 seconds
# Measured by xtime.rb: 5.08s, 3882.8Mb

#############################################################
# Multi-threaded - OpenMP
# Note: there are probably multicore cache invalidation issues that slow down multithreading.

# Measurement on i5-5257U (Dual core mobile Broadwell 2.7Ghz)
# Arraymancer 0.4.0 and Nim devel after perf regression fix 88cf6573e04bd7ee8762aa336460b9748f0d4644

# Arraymancer Euler solve - time taken: 17.7961540222168 seconds
# Measured by xtime.rb: 18.11s, 3719.5Mb
