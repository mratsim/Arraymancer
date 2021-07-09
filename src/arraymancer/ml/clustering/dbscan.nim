import deques
import sequtils

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
      j = peekLast(stack)
      popLast(stack)
    label_num += 1
  result = labels

proc dbscan*[T: SomeFloat](X: Tensor[T], eps: float, minSamples: int,
                           metric: typedesc[Manhattan | Euclidean | Minkowski | Jaccard],
                           p = 2.0): seq[int] =
  when metric is Minkowski:
    let neighborhoods = nearestNeighbors(X, eps, metric, p = p)
  else:
    let neighborhoods = nearestNeighbors(X, eps, metric)
  let isCore = findCorePoints(neighborhoods, minSamples)
  result = assignLabels(isCore, neighborhoods)
