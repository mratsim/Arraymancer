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


proc f(T: Tensor[float]): Tensor[float] {.noInit.}=
  map3_inline(T[_, 0..^3], T[_, 1..^2], T[_, 2..^1]):
    a_dz2 * (x - 2.0 * y + z)

proc eulerSolve(Ts: var Tensor[float]) =
  for t in 0 ..< timeSteps-1:
    var slice1 = Ts[t+1, 1..^2]
    apply3_inline(slice1, Ts[t, 1..^2], f(Ts[t, _])):
      y + dt * z

    var slice2 = Ts[t+1, ^1]
    apply2_inline(slice2, Ts[t+1, ^2]):
      y


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


# Measurement on i5-5257U (Dual core mobile Broadwell 2.7Ghz)

# Arraymancer Euler solve - time taken: 8.375565 seconds
# xtime.rb - 10.22s, 3854.9Mb

# OpenMP slows things by a lot, probably due to false sharing with the non-contiguous slices.

# Measurement on i7-970 (Hexa core 3.2GHz)
# 42.0060796609176
# 34.83783780774945
# 29.89741051712985
# Arraymancer Euler solve - time taken: 5.060707 seconds
# 5.08s, 3882.8Mb

###################################
# Optimized version with no temporaries

# Single-threaded i5-5257U (Dual core mobile Broadwell 2.7Ghz)
# Arraymancer Euler solve - time taken: 6.646692 seconds
# 7.47s, 3037.2Mb

# Measurement on i7-970 (Hexa core 3.2GHz)
# 42.0060796609176
# 34.83783780774945
# 29.89741051712985
# Arraymancer Euler solve - time taken: 2.085845 seconds
# 2.10s, 3882.8Mb
