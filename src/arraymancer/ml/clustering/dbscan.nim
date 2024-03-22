import std / [deques, sequtils]

import ../../spatial/[distances, neighbors],
       ../../tensor

export distances

proc findCorePoints(neighborhoods: seq[Tensor[int]], minSamples: int): seq[bool] =
  ## Iterates over the neighborhoods of all points. Each element of `neighborhoods` represents
  ## the indices of points that are within a distance of `eps` (see `dbscan` proc) under
  ## the chosen metric.
  ##
  ## If a point has more than `minSamples` in its neighborhood, it is considered a "core" point.
  let nSamples = neighborhoods.len
  result = newSeq[bool](nSamples) # init to `false` by default
  for i in 0 .. neighborhoods.high:
    if neighborhoods[i].size >= minSamples:
      result[i] = true

proc assignLabels(is_core: seq[bool], neighborhoods: seq[Tensor[int]]): seq[int] =
  ## Assigns the cluster labels to all input points based on their neighborhood
  ## and whether they are core points as determined in `findCorePoints`.
  let nSamples = len(neighborhoods)
  var
    label_num, j, v: int
    labels = repeat(-1, nSamples)
    neighbors: Tensor[int]
    stack = initDeque[int]()

  for i in 0 ..< nSamples:
    if labels[i] != -1 or is_core[i] == false:
      continue
    j = i
    while true:
      if labels[j] == -1:
        labels[j] = label_num
        if is_core[j]:
          neighbors = neighborhoods[j]
          for j in 0 ..< neighbors.size:
            v = neighbors[j]
            if labels[v] == -1:
              stack.addLast(v)
      if len(stack) == 0:
        break
      j = popLast(stack)
    label_num += 1
  result = labels

proc dbscan*[T: SomeFloat](X: Tensor[T], eps: float, minSamples: int,
                           metric: typedesc[AnyMetric] = Euclidean,
                           p = 2.0): seq[int] =
  ## Performs `DBSCAN` clustering on the input data `X`. `X` needs to be a tensor
  ## of rank 2 with the following shape:
  ##
  ## - `[n_observations, n_dimensions]`
  ##
  ## so that we have `n_observations` points that each have a dimensionality of
  ## `n_dimensions` (or sometimes called number of features).
  ##
  ## `eps` is the radius in which we search for neighbors around each point using
  ## the give `metric`.
  ##
  ## `minSamples` is the minimum number of elements that need to be in the search
  ## radius `eps` to consider a set of points a proto-cluster (the "core points"),
  ## from which to compute the final clusters.
  ##
  ## If we use the Minkowski metric, `p` is the power to use in it. Otherwise
  ## the value is ignored.
  let neighborhoods = nearestNeighbors(X, eps, metric, p = p)
  let isCore = findCorePoints(neighborhoods, minSamples)
  result = assignLabels(isCore, neighborhoods)
