import math, times, strformat

const
  dz = 0.01
  z = 100
  spaceSteps = int(z / dz)
  timeSteps = 10000
  dt = 0.12 / timeSteps
  alpha = 2.0
  startingTemp = 30.0
  oscillations = 20.0


proc f(T: array[spaceSteps, float]): array[spaceSteps-2, float] =
  for i in 0 .. spaceSteps-3:
    result[i] = alpha * (T[i] - 2.0 * T[i+1] + T[i+2]) / dz^2


proc eulerSolve(Ts: var array[timeSteps, array[spaceSteps, float]]) =
  for t in 0 ..< timeSteps-1:
    let
      k = f(Ts[t])
    for i in 1 .. spaceSteps-2:
      Ts[t+1][i] = Ts[t][i] + dt * k[i-1]
    Ts[t+1][spaceSteps-1] = Ts[t+1][spaceSteps-2]

let start = cpuTime()

# TODO replace by a main proc
# once https://github.com/nim-lang/Nim/issues/7349 is solved
var Ts: array[timeSteps, array[spaceSteps, float]]

for t in 0 ..< timeSteps:
  Ts[t][0] = startingTemp - oscillations * sin(2.0 * PI * t.float / timeSteps)
  for s in 1 ..< spaceSteps:
    Ts[t][s] = startingTemp

Ts.eulerSolve()

let stop = cpuTime()

let elapsed = stop - start
echo &"Arraymancer Euler solve - time taken: {elapsed} seconds"

# Measurement on i5-5257U (Dual core mobile Broadwell 2.7Ghz)
# Arraymancer Euler solve - time taken: 0.8557929999999999 seconds
# 0.92s, 764.0Mb (as measured by xtime.rb)
