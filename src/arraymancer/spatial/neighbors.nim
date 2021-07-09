import algorithm
import sequtils

import ../tensor
import ./distances
import kdtree

proc neighborsFromDistanceMatrix[T](X: Tensor[T], eps: float): seq[int] =
  ## Computes the nearest neighbors from a naive distance matrix between all points
  ## in `X` that are within a distance of `eps`.
  result = newSeqOfCap[int](X.shape[0])
  for j in argsort(X.transpose.squeeze, toCopy = true):
    if X[j, 0] > eps:
      break
    result.add(j)

proc nearestNeighbors*[T](X: Tensor[T], eps: float, metric: typedesc[AnyMetric],
                          p = 2.0,
                          useNaiveNearestNeighbor: static bool = false): seq[Tensor[int]] =
  ## Computes nearest neighbors of all points in `X` that are within a distance of `eps`
  ## under the given `metric`.
  ##
  ## The input tensor `X` must be of rank 2 and contain data as:
  ##
  ## - `[n_observations, n_dimensions]`
  ##
  ## If the Minkowski metric is used `p` corresponds to the power used for the metric.
  ##
  ## If `useNaiveNearestNeighbor` is set to `true` a naive nearest neighbor computation is
  ## performed. This is not advised, as it is significantly slower than the default approach
  ## using a k-d tree.
  # TODO: extend k-d tree to allow usage of all our metrics defined in `distances.nim`! Otherwise
  # the metric given is useless for the k-d tree case.
  when not useNaiveNearestNeighbor:
    let kd = kdtree(X)
    for v in axis(X, 0):
      let (dist, idxs) = kd.query_ball_point(v.squeeze, radius = eps)
      result.add idxs
  else:
    when metric is Minkowski:
      let distances = distanceMatrix(metric, X, X, p = 2)
    else:
      let distances = distanceMatrix(metric, X, X)
    result = newSeq[seq[int]](X.shape[0])
    var idx = 0
    for v in axis(distances, 0):
      result[idx] = neighborsFromDistanceMatrix(v, eps)
      inc idx
