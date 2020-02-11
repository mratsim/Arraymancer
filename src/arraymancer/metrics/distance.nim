import math
import sets
import sequtils

import ../../tensor/tensor


type
  PairwiseDist* = ref object of RootObj
  Euclidean* = ref object of PairwiseDist
  Manhattan* = ref object of PairwiseDist
  Minkowski* = ref object of PairwiseDist
    p*: int
  Jaccard* = ref object of PairwiseDist


method computeDistance(metric: PairwiseDist, v, w: Tensor[float]): float {.base.} = 
  return 0.0


method computeDistance(metric: Manhattan, v, w: Tensor[float]): float = 
  return sum(abs(v -. w))


method computeDistance(metric: Minkowski, v, w: Tensor[float]): float = 
  let p = float(metric.p)
  let d = abs(v -. w)
  result = pow(sum(mapIt(d, pow(it, p))), 1/p)


method computeDistance(metric: Jaccard, v, w: Tensor[float]): float = 
  let sx = toHashSet(toSeq(v))
  let sy = toHashSet(toSeq(w))
  result = len(intersection(sx, sy)) / len(union(sx, sy))


method pairwise*(metric: PairwiseDist, x, y: Tensor[float]): Tensor[float] {.base.} =
  let n_obs = x.shape[1]
  result = newTensor[float](n_obs, n_obs)
  for i, v in enumerateAxis(x, 1):
    for j, w in enumerateAxis(x, 1):
      if j < i:
        result[i,j] = result[j,i]
        continue
      result[i,j] = computeDistance(metric, v, w)
 

method pairwise*(metric: Euclidean, x, y: Tensor[float]): Tensor[float] =
  let xx = sum(x *. x, axis=0)
  let yy = transpose(sum(y *. y, axis=0))
  result = sqrt(-2.0 * transpose(x) * y +. xx +. yy)
