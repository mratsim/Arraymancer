import math
import sets
import sequtils

import ../tensor

type
  Euclidean* = object
  Manhattan* = object
  Minkowski* = object
  Jaccard* = object

  AnyMetric* = Euclidean | Manhattan | Minkowski | Jaccard

## Calculates the euclidean distance
#  ## Inputs:
#  ##  - u: a tensor of shape (nb samples, nb features)
#  ##  - v: a tensor of shape (nb samples, nb features)
#  ##  - squared: whether or not to square the result
#  ##
#  ## sqrt( (u - v) dot (u - v) )
#  ## which can be reformulated
#  ## sqrt((u dot u) - 2 * (u dot v) + (v dot v))
#  ##
#  ## Returns:
#  ##  - A tensor of shape (nb samples)
proc toHashSet[T](t: Tensor[T]): HashSet[T] =
  result = initHashSet[T](t.size)
  for x in t:
    result.incl x

proc distance*(metric: typedesc[Manhattan], v, w: Tensor[float]): float =
  ## Computes the Manhattan distance between points `v` and `w`. Both need to
  ## be rank 1 tensors with `k` elements, where `k` is the dimensionality
  ## of the points.
  ##
  ## The Manhattan metric is defined as:
  ##
  ## ``d = Σ_i | v_i - w_i |``
  assert v.rank == 1
  assert w.rank == 1
  result = sum(abs(v -. w))

proc distance*(metric: typedesc[Minkowski], v, w: Tensor[float], p = 2.0,
               squared: static bool = false): float =
  ## Computes the Minkowski distance between points `v` and `w`. Both need to
  ## be rank 1 tensors with `k` elements, where `k` is the dimensionality
  ## of the points.
  ##
  ## The Minkowski metric is defined as:
  ##
  ## ``d = ( Σ_i | v_i - w_i |^p )^(1/p)``
  ##
  ## Thus, it reduces to the Manhattan distance for `p = 1` and the Euclidean metric
  ## for `p = 2`.
  ##
  ## If `squared` is true returns the `p`-th power of the metric. For the Euclidean case
  ## this is the square of the distance, hence the name.
  assert v.squeeze.rank == 1
  assert w.squeeze.rank == 1
  when squared:
    result = sum( abs(v -. w).map_inline(pow(x, p)) )
  else:
    result = pow( sum( abs(v -. w).map_inline(pow(x, p)) ), 1.0 / p )

proc distance*(metric: typedesc[Euclidean], v, w: Tensor[float], squared: static bool = false): float =
  ## Computes the Euclidean distance between points `v` and `w`. Both need to
  ## be rank 1 tensors with `k` elements, where `k` is the dimensionality
  ## of the points.
  ##
  ## The Euclidean metric is defined as:
  ##
  ## ``d = ( Σ_i | v_i - w_i |^2 )^(1/2)``
  ##
  ## If `squared` is true returns the square of the distance
  assert v.squeeze.rank == 1
  assert w.squeeze.rank == 1
  # Note: possibly faster by writing `let uv = u -. v; dot(uv, uv);` ?
  result = Minkowski.distance(v, w, p = 2.0, squared = squared)

proc distance*(metric: typedesc[Jaccard], v, w: Tensor[float]): float =
  ## Computes the Jaccard distance between points `v` and `w`. Both need to
  ## be rank 1 tensors with `k` elements, where `k` is the dimensionality
  ## of the points.
  ##
  ## The Jaccard distance is defined as:
  ##
  ## ``d = 1 - J(A, B) = ( | A ∪ B | - | A ∩ B | ) / ( | A ∪ B | )``
  # Note: doesn't this make the most sense for non floating point tensors?
  let sx = toHashSet(v)
  let sy = toHashSet(w)
  result = 1 - len(intersection(sx, sy)) / len(union(sx, sy))

proc pairwiseDistances*(metric: typedesc[AnyMetric],
                        x, y: Tensor[float],
                        p = 2.0,
                        squared: static bool = false): Tensor[float] =
  ## Computes all distances between all pairs in `x` and `y`. That is if
  ## `x` and `y` are rank 2 tensors of each:
  ##
  ## - `[n_observations, n_dimensions]`
  ##
  ## we compute the distance between each observation `x_i` and `y_i`.
  ##
  ## The first argument is the metric to compute the distance under. If the Minkowski metric
  ## is selected the power `p` is used.
  ##
  ## If `squared` is true and we are computing under a Minkowski or Euclidean metric, we return
  ## the `p`-th power of the distances.
  ##
  ## Result is a tensor of rank 1, with one element for each distance.
  assert x.rank == 2
  assert y.rank == 2
  let n_obs = x.shape[0]
  result = newTensorUninit[float](n_obs)
  for idx in 0 ..< n_obs:
    when metric is Minkowski:
      result[idx] = distance(metric, x[idx, _].squeeze, y[idx, _].squeeze,
                             p = p, squared = squared)
    elif metric is Euclidean:
      result[idx] = distance(metric, x[idx, _].squeeze, y[idx, _].squeeze,
                             squared = squared)
    else:
      result[idx] = distance(metric, x[idx, _].squeeze, y[idx, _].squeeze)

proc distanceMatrix*(metric: typedesc[AnyMetric],
                     x, y: Tensor[float],
                     p = 2.0,
                     squared: static bool = false): Tensor[float] =
  ## Computes the distance matrix between all points in `x` and `y`.
  ## `x` and `y` need to be tensors of rank 2 with:
  ##
  ## - `[n_observations, n_dimensions]`
  ##
  ## The first argument is the metric to compute the distance under. If the Minkowski metric
  ## is selected the power `p` is used.
  ##
  ## If `squared` is true and we are computing under a Minkowski or Euclidean metric, we return
  ## the `p`-th power of the distances.
  ##
  ## Result is a tensor of rank 2, a symmetric matrix where element `(i, j)` is the distance
  ## between `x_i` and `y_j`.
  assert x.rank == 2
  assert y.rank == 2
  let n_obs = x.shape[0]
  result = newTensorUninit[float](n_obs, n_obs)
  for i, v in enumerateAxis(x, 0):
    for j, w in enumerateAxis(y, 0):
      if j < i:
        result[i,j] = result[j,i]
        continue
      when metric is Minkowski:
        result[i,j] = distance(metric, v, w, p = p, squared = squared)
      elif metric is Euclidean:
        result[i,j] = distance(metric, v, w, squared = squared)
      else:
        result[i,j] = distance(metric, v, w)
