import algorithm
import sequtils

import ../../tensor/tensor
import ../metrics/distance


proc argsort[T: SomeFloat](x: seq[T]): seq[int] =
  result = toSeq(0..high(x))
  sort(result, proc(i, j:int): int = cmp(x[i], x[j]))


proc neighborsFromDistanceMatrix[T](X: Tensor[T], eps: float): seq[int] =
  for j in argsort(toSeq(X)):
    if X[j, 0] > eps:
      break
    result.add(j)


proc simpleNearestNeighbors*[T](X: Tensor[T], eps:float, metric: PairwiseDist): seq[seq[int]] =
  let distances = pairwise(metric, X, X)
  for v in axis(distances, 1):
    result.add(neighborsFromDistanceMatrix(v, eps))
