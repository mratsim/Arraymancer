import math, times, strformat

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


var ke: array[spaceSteps-2, float]


proc f(k: var array[spaceSteps-2, float], T: array[spaceSteps, float]) =
  for i in 0 .. spaceSteps-3:
    k[i] = a_dz2 * (T[i] - 2.0 * T[i+1] + T[i+2])


proc eulerSolve(Ts: var seq[array[spaceSteps, float]]) =
  for t in 0 ..< timeSteps-1:
    f(ke, Ts[t])
    for i in 1 .. spaceSteps-2:
      Ts[t+1][i] = Ts[t][i] + dt * ke[i-1]
    Ts[t+1][spaceSteps-1] = Ts[t+1][spaceSteps-2]


proc main() =
  var Ts = newSeq[array[spaceSteps, float]](timeSteps)

  for t in 0 ..< timeSteps:
    Ts[t][0] = startingTemp - oscillations * sin(2.0 * PI * t.float / timeSteps)
    for s in 1 ..< spaceSteps:
      Ts[t][s] = startingTemp

  Ts.eulerSolve()

let start = cpuTime()
main()
let stop = cpuTime()

let elapsed = stop - start
echo &"Native array Euler solve - time taken: {elapsed} seconds"


# Measurement on i7-970 (Hexa core 3.2GHz)
# Native array Euler solve - time taken: 2.732873 seconds
# 2.87s, 3816.5Mb
