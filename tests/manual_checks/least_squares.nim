
import ../../src/arraymancer

block:
  let a = [
    [ 0.12f,  -8.19,   7.69,  -2.26,  -4.71],
    [-6.91f,   2.22,  -5.12,  -9.08,   9.96],
    [-3.33f,  -8.94,  -6.72,  -4.40,  -9.98],
    [ 3.97f,   3.33,  -2.74,  -7.92,  -3.20]
  ].toTensor

  let b = [
    [ 7.30f,   0.47f,  -6.28f],
    [ 1.33f,   6.58f,  -3.42f],
    [ 2.68f,  -1.71f,   3.46f],
    [-9.62f,  -0.79f,   0.41f]
  ].toTensor

  let sol = least_squares_solver(a, b)
  echo sol

block:
  let A = [
    [0f, 1f],
    [1f, 1f],
    [2f, 1f],
    [3f, 1f]
  ].toTensor

  let y = [-1f, 0.2, 0.9, 2.1].toTensor

  let sol = least_squares_solver(A, y)
  echo sol
