import deques
import sequtils

import ../../spatial/[distance, neighbors],
       ../../tensor

export distances

proc findCorePoints(neighborhoods: seq[seq[int]], minSamples: int): seq[bool] =
  let nSamples = len(neighborhoods)
  result = repeat(false, nSamples) 
  for i in 0..high(neighborhoods):
    if len(neighborhoods[i]) >= minSamples:
      result[i] = true


proc assignLabels(is_core: seq[bool], neighborhoods: seq[seq[int]]): seq[int] =
  let nSamples = len(neighborhoods)
  var
    label_num, j, v: int
    labels = repeat(-1, nSamples)
    neighbors: seq[int]
    stack = initDeque[int]()

  for i in 0..<nSamples:
    if labels[i] != -1 or is_core[i] == false:
      continue
    j = i
    while true:
      if labels[j] == -1:
        labels[j] = label_num
        if is_core[j]:
          neighbors = neighborhoods[j]
          for j in 0..high(neighbors):
            v = neighbors[j]
            if labels[v] == -1:
              stack.addLast(v)
      if len(stack) == 0:
        break
      j = peekLast(stack)
      popLast(stack)
    label_num += 1
  return labels


proc dbscan*[T: SomeFloat](X: Tensor[T], eps: float, minSamples: int, metric: PairwiseDist): seq[int] =
  let neighborhoods = simpleNearestNeighbors(X, eps, metric)
  let is_core = findCorePoints(neighborhoods, minSamples)
  result = assignLabels(is_core, neighborhoods)