import math
import sets

import ../tensor

type
  Euclidean* = object
  Manhattan* = object
  Minkowski* = object
  Jaccard* = object
  CustomMetric* = object

  AnyMetric* = Euclidean | Manhattan | Minkowski | Jaccard | CustomMetric

when (NimMajor, NimMinor, NimPatch) < (1, 4, 0):
  # have to export sets for 1.0, because `bind` didn't exist apparently
  export sets

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
  assert v.squeeze.rank == 1
  assert w.squeeze.rank == 1
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
  if classify(p) == fcInf:
    result = max(abs(v -. w))
  elif p == 1:
    result = sum(abs(v -. w))
  else:
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
  #result = Minkowski.distance(v, w, p = 2.0, squared = squared)
  ## NOTE: this is the branch used in the kd-tree. It's very performance critical there,
  ## hence we use this simple manual code (benchmarked to be more than 2x faster than
  ## via a 'higher order' approach).
  ## DBSCAN clustering test (11,000 points)
  ## - debug mode, old branch: 98.5s
  ## - debug mode, this branch: 50s
  ## - danger mode, old branch: 6.3s
  ## - danger mode, this branch: 2.8s
  when squared:
    if v.is_C_contiguous and w.is_C_contiguous:
      result = 0.0
      var tmp = 0.0
      let vBuf = v.toUnsafeView()
      let wBuf = w.toUnsafeView()
      for idx in 0 ..< v.size:
        # Use `atIndex` so that this also works for rank 2 tensors with `[1, N]` size, as this is
        # what we get from `pairwiseDistance` due to not squeezing the dimensions anymore.
        tmp = vBuf[idx] - wBuf[idx] # no need for abs, as we square
        result += tmp*tmp
    else: # Fall back to broadcasting implementation which handles non contiguous data
      result = sum( abs(v -. w).map_inline(x * x) )
  else:
    result = sqrt( sum( abs(v -. w).map_inline(x * x) ) )

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
  # bind items for sets here to make it work w/o sets imported
  when (NimMajor, NimMinor, NimPatch) >= (1, 4, 0):
    # can only `bind` from 1.2, as it didn't exist in 1.0?
    bind sets.items
  result = 1 - intersection(sx, sy).card.float / union(sx, sy).card.float

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
  ## One of the arguments may have only 1 observation and thus be of shape
  ## `[1, n_dimensions]`. In this case all distances between this point and
  ## all in the other input will be computed so that the result is always of
  ## shape `[n_observations]`.
  ## If one input has only shape `[n_dimensions]` it is unsqueezed to `[1, n_dimensions]`.
  ##
  ## The first argument is the metric to compute the distance under. If the Minkowski metric
  ## is selected the power `p` is used.
  ##
  ## If `squared` is true and we are computing under a Minkowski or Euclidean metric, we return
  ## the `p`-th power of the distances.
  ##
  ## Result is a tensor of rank 1, with one element for each distance.
  let n_obs = max(x.shape[0], y.shape[0])
  result = newTensorUninit[float](n_obs)

  if x.rank == y.rank and x.shape[0] == y.shape[0]:
    for idx in 0 ..< n_obs:
      when metric is Minkowski:
        result[idx] = Minkowski.distance(x[idx, _], y[idx, _],
                                         p = p, squared = squared)
      elif metric is Euclidean:
        result[idx] = Euclidean.distance(x[idx, _], y[idx, _],
                                         squared = squared)
      else:
        result[idx] = metric.distance(x[idx, _], y[idx, _])
  else:
    # determine which is one is 1 along n_observations
    let nx = if x.rank == 2 and x.shape[0] == n_obs: x else: y
    var ny = if x.rank == 2 and x.shape[0] == n_obs: y else: x
    # in this case compute distance between all `nx` and single `ny`
    if ny.rank == 1: # unsqueeze to have both rank 2
      ny = ny.unsqueeze(0)
    var idx = 0
    for ax in axis(nx, 0):
      when metric is Minkowski:
        result[idx] = Minkowski.distance(ax, ny,
                                         p = p, squared = squared)
      elif metric is Euclidean:
        result[idx] = Euclidean.distance(ax, ny,
                                         squared = squared)
      else:
        result[idx] = metric.distance(ax, ny)
      inc idx

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
