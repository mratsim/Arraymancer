import algorithm
import sequtils

import ../tensor
import ./distance

proc neighborsFromDistanceMatrix[T](X: Tensor[T], eps: float): seq[int] =
  for j in argsort(X, toCopy = true):
    if X[j, 0] > eps:
      break
    result.add(j)

proc simpleNearestNeighbors*[T](X: Tensor[T], eps:float, metric: PairwiseDist): seq[seq[int]] =
  let distances = pairwise(metric, X, X)
  for v in axis(distances, 1):
    result.add(neighborsFromDistanceMatrix(v, eps))
