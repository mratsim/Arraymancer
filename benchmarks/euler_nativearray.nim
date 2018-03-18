import math, times, strformat

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
    Ts[0][s] = startingTemp

  Ts.eulerSolve()
  echo Ts[45_000][10]
  echo Ts[45_000][100]
  echo Ts[45_000][500]

let start = cpuTime()
main()
let stop = cpuTime()

let elapsed = stop - start
echo &"Native array Euler solve - time taken: {elapsed} seconds"


# Measurement on i7-970 (Hexa core 3.2GHz)
# 42.0060796609176
# 34.83783780774945
# 29.89741051712985
# Native array Euler solve - time taken: 1.59722 seconds
# 1.61s, 3816.6Mb
