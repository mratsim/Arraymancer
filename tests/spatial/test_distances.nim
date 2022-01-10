import ../../src/arraymancer
import unittest, random, sequtils, algorithm

proc `=~=`(x, y: float, eps = 1e-6): bool =
  ## simple float compare with epsilon to not have to worry about perfect scipy / nim
  ## comparisons
  result = abs(x - y) < eps

template `=~=`(x, y: Tensor[float], eps = 1e-6): untyped =
  ## simple float compare with epsilon to not have to worry about perfect scipy / nim
  ## comparisons
  check x.size == y.size
  for i in 0 ..< x.size:
    check x[i] =~= y[i]

suite "Distances: Manhattan distance":
  test "Manhattan distance - Individual points":
    let t = [1.0, 2.0].toTensor
    let u = [0.0, 0.0].toTensor
    check Manhattan.distance(t, u) == 3.0

  test "Manhattan distance - Pairwise of multiple points + single point":
    let t = [[1.0, 2.0], [3.0, 5.0]].toTensor
    let u = [0.0, 0.0].toTensor
    check Manhattan.pairwiseDistances(t, u) == [3.0, 8.0].toTensor
    # order does not matter
    check Manhattan.pairwiseDistances(u, t) == [3.0, 8.0].toTensor

  test "Manhattan distance - Pairwise of multiple points":
    let t = [[1.0, 2.0], [3.0, 5.0]].toTensor
    let u = [[0.0, 0.0], [1.0, 2.0]].toTensor
    check Manhattan.pairwiseDistances(t, u) == [3.0, 5.0].toTensor

suite "Distances: Euclidean distance":
  test "Euclidean distance - Individual points":
    let t = [3.0, 4.0].toTensor
    let u = [0.0, 0.0].toTensor
    check Euclidean.distance(t, u) == 5.0

  test "Euclidean distance - Individual points, squared":
    let t = [3.0, 4.0].toTensor
    let u = [0.0, 0.0].toTensor
    check Euclidean.distance(t, u, squared = true) == 25.0

  test "Euclidean distance - Pairwise of multiple points + single point":
    let t = [[3.0, 4.0], [5.0, 5.0]].toTensor
    let u = [0.0, 0.0].toTensor
    Euclidean.pairwiseDistances(t, u) =~= [5.0, 7.07106781187].toTensor
    # order does not matter
    Euclidean.pairwiseDistances(u, t) =~= [5.0, 7.07106781187].toTensor

  test "Euclidean distance - Pairwise of multiple points":
    let t = [[3.0, 4.0], [4.0, 6.0]].toTensor
    let u = [[0.0, 0.0], [1.0, 2.0]].toTensor
    check Euclidean.pairwiseDistances(t, u) == [5.0, 5.0].toTensor

suite "Distances: Minkowski distance":
  test "Minkowski distance - Individual points, p = 1":
    let t = [3.0, 4.0].toTensor
    let u = [0.0, 0.0].toTensor
    check Manhattan.distance(t, u) == Minkowski.distance(t, u, p = 1.0)

  test "Minkowski distance - Individual points, p = 2":
    let t = [3.0, 4.0].toTensor
    let u = [0.0, 0.0].toTensor
    check Euclidean.distance(t, u) == Minkowski.distance(t, u, p = 2.0)

  test "Minkowski distance - Individual points, p = 2, squared":
    let t = [3.0, 4.0].toTensor
    let u = [0.0, 0.0].toTensor
    check Euclidean.distance(t, u, squared = true) == Minkowski.distance(t, u, p = 2.0, squared = true)

  test "Minkowski distance - Pairwise of multiple points + single point":
    let t = [[3.0, 4.0], [5.0, 5.0]].toTensor
    let u = [0.0, 0.0].toTensor
    Minkowski.pairwiseDistances(t, u, p = 2.0) =~= [5.0, 7.07106781187].toTensor
    # order does not matter
    Minkowski.pairwiseDistances(u, t, p = 2.0) =~= [5.0, 7.07106781187].toTensor

  test "Minkowski distance - Pairwise of multiple points":
    let t = [[3.0, 4.0], [4.0, 6.0]].toTensor
    let u = [[0.0, 0.0], [1.0, 2.0]].toTensor
    check Minkowski.pairwiseDistances(t, u, p = 2.0) == [5.0, 5.0].toTensor

suite "Distances: Jaccard distance":
  test "Jaccard distance - Individual points":
    block:
      let t = [3.0, 4.0].toTensor
      let u = [3.0, 4.0].toTensor
      check Jaccard.distance(t, u) == 0.0
    block:
      let t = [3.0, 5.0].toTensor
      let u = [3.0, 4.0].toTensor
      check Jaccard.distance(t, u) =~= 2.0 / 3.0

  test "Jaccard distance - Pairwise of multiple points + single point":
    let t = [[3.0, 4.0], [3.0, 5.0]].toTensor
    let u = [3.0, 4.0].toTensor
    Jaccard.pairwiseDistances(t, u) =~= [0.0, 2.0 / 3.0].toTensor
    # order does not matter
    Jaccard.pairwiseDistances(u, t) =~= [0.0, 2.0 / 3.0].toTensor

  test "Jaccard distance - Pairwise of multiple points":
    let t = [[3.0, 4.0], [3.0, 5.0]].toTensor
    let u = [[3.0, 4.0], [3.0, 4.0]].toTensor
    Jaccard.pairwiseDistances(t, u) =~= [0.0, 2.0 / 3.0].toTensor

suite "Distances: `CustomMetric`":
  let a = [1,2,3].toTensor
  let b = [4,5,6].toTensor
  test "CustomMetric of a constant value":
    proc distance(_: typedesc[CustomMetric], v, w: Tensor[int]): int =
      result = 100
    check CustomMetric.distance(a, b) == 100

  test "CustomMetric that is euclidean":
    proc distance(_: typedesc[CustomMetric], v, w: Tensor[int]): float =
      result = Euclidean.distance(v.asType(float), w.asType(float))

    check CustomMetric.distance(a, b) == Euclidean.distance(a.asType(float), b.asType(float))
