import math, times, strformat
import ../src/arraymancer

const
  dz = 0.01
  z = 100
  spaceSteps = int(z / dz)
  timeSteps = 50000
  dt = 0.12 / timeSteps
  alpha = 2.0
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

let start = cpuTime()
main()
let stop = cpuTime()

let elapsed = stop - start
echo &"Arraymancer Euler solve - time taken: {elapsed} seconds"


# Measurement on i7-970 (Hexa core 3.2GHz)
# Arraymancer Euler solve - time taken: 5.857952 seconds
# 6.01s, 3882.8Mb
