#import ../tensor/tensor
import arraymancer
import sequtils, math, heapqueue, sugar

type
  TreeNodeKind = enum
    tnLeaf, tnInner

  Node[T] = ref object
    #level: int
    idx: Tensor[int]
    split_dim: int
    split: float
    children: int
    data: Tensor[T]
    indices: Tensor[int]
    case kind: TreeNodeKind
    of tnInner:
      lesser: Node[T]
      greater: Node[T]
    of tnLeaf: discard

  KDTree[T] = ref object
    data: Tensor[T]
    leafsize: int ## default 16
    m: int ## dimension of a single data point
    n: int ## number of data points
    maxes: Tensor[T] ## maximum values along each dimension of n data points
    mins: Tensor[T] ## minimum values along each dimension of n data points
    tree: Node[T] ## the root node of the tree
    size: int ## number of nodes in the tree

proc `<`[T](n1, n2: Node[T]): bool =
  ## now this is a little sketchy, but (from what I understand) follows what numpy does.
  ## Comparisons of Nodes is done via the `id` of each object, which for CPython is
  ## just the address if I understand correctly
  ## https://github.com/scipy/scipy/blob/master/scipy/spatial/kdtree.py#L251-L252
  result = cast[int](n1.unsafeAddr) < cast[int](n2.unsafeAddr)

proc `<`[T](s1, s2: seq[T]): bool =
  ## just an internal comparison of two seqs, which assumes that the order of two
  ## seqs matters.
  doAssert s1.len == s2.len
  result = false
  for i in 0 ..< s1.len:
    if s1[i] == s2[i]:
      # may still be decided
      continue
    elif s1[i] < s2[i]:
      return true
    elif s1[i] > s2[i]:
      return false

proc nonzero[T](t: Tensor[T]): seq[seq[int]] =
  ## returns the indices, which are non zero along `axis` as a `seq[seq[int]]`.
  ## One `seq[int]` for each dimension of `t`, which contains the indices
  ## of nonzero elements along that axis
  let mask = t .!= 0.T
  result = newSeqWith(t.shape.len, newSeqOfCap[int](t.size))
  var i = 0
  var ax = 0
  for idx, x in mask:
    if x:
      ax = 0
      for j in idx:
        result[ax].add j
        inc ax
    inc i

proc allEqual[T](t: Tensor[T], val: T): bool =
  ## checks if all elements of `t` are `val`
  result = true
  for x in t:
    if x != val:
      return false

proc fancyIndex[T](t: Tensor[T], idx: Tensor[int]): Tensor[T] =
  #result = newTensorUninit(idx.size)
  result = t.index_select(axis = 0, idx)

proc build[T](tree: KDTree[T],
              idx: Tensor[int],
              #startIdx, endIdx: int
              maxes, mins: Tensor[T]): Node[T] =
              #useMedian: static bool,
              #createCompact: static bool) =
  echo "BUILD!"
  ## recursively build the KD tree
  if idx.size <= tree.leafSize:
    result = Node[T](kind: tnLeaf,
                     idx: idx,
                     children: idx.size)
  else:
    var data = tree.data.fancyIndex(idx)
    let d = argmax((maxes .- mins).squeeze, axis = 0)[0]
    let maxVal = maxes[d]
    let minVal = mins[d]
    if maxVal == minVal:
      return Node[T](kind: tnLeaf,
                     idx: idx,
                     children: idx.size)
    data = squeeze(data[_, d])

    # sliding midpoint rule
    var split = (maxVal + minVal) / 2.0
    var lessIdx = toTensor nonzero(data .<= split)[0]
    var greaterIdx = toTensor nonzero(data .> split)[0]
    if lessIdx.size == 0:
      split = min(data)
      lessIdx = toTensor nonzero(data .<= split)[0]
      greaterIdx = toTensor nonzero(data .> split)[0]
    if greaterIdx.size == 0:
      split = max(data)
      lessIdx = toTensor nonzero(data .< split)[0]
      greaterIdx = toTensor nonzero(data .>= split)[0]
    if lessIdx.size == 0:
      # still zero, all must have same value
      if not allEqual(data, data[0]):
        raise newException(ValueError, "Bad data: " & $data)
      split = data[0]
      lessIdx = arange(data.size - 1)
      greaterIdx = toTensor @[data.size - 1]

    var lessmaxes = maxes.clone()
    lessmaxes[d] = split
    var greatermins = mins.clone()
    greatermins[d] = split

    result = Node[T](kind: tnInner,
                     split_dim: d,
                     split: split,
                     lesser: tree.build(idx.fancyIndex(lessIdx), lessmaxes, mins),
                     greater: tree.build(idx.fancyIndex(greaterIdx), maxes, greatermins))

proc buildKdTree[T](tree: var KDTree[T],
                    startIdx: Tensor[int],
                    useMedian: static bool,
                    createCompact: static bool) =
  tree.tree = tree.build(idx = startIdx,
                         maxes = tree.maxes,
                         mins = tree.mins)

proc kdTree[T](data: Tensor[T], ## data must be 2D tensor (n, m)
               leafSize = 16,
               compactNodes: static bool = true, ## default true, compacter tree, but longer build time
               copyData = true, ## default false, data is not copied
               ## default true, split at median instead of mid point
               ## smaller trie, faster queries, longer build time
               balancedTree: static bool = true): KDTree[T] =
  result = new(KDTree[T])
  result.data = if copyData: data.clone() else: data
  if data.shape.len != 2:
    raise newException(ValueError, "Data must have 2 dimensions!")
  result.n = data.shape[0]
  result.m = data.shape[1]
  result.leafsize = leafsize
  doAssert leafsize > 0, "leafSize must be at least 1!"

  result.maxes = data.max(axis = 0).squeeze
  result.mins = data.min(axis = 0).squeeze
  # result.indices =

  result.buildKdTree(arange[int](result.n),
                     useMedian = balancedTree,
                     createCompact = compactNodes)

func minkowski_distance_p[T](x, y: Tensor[T], p = 2.0): Tensor[T] =
  let ax = x.shape.len - 1
  if classify(p) == fcInf:
    result = max(abs(y .- x), axis = ax)
  elif p == 1:
    result = sum(abs(y .- x), axis = ax)
  else:
    result = sum(abs(y .- x).map_inline(pow(x, p)), axis = ax)

func minkowski_distance[T](x, y: Tensor[T], p = 2.0): Tensor[T] =
  if classify(p) == fcInf or p == 1:
    result = minkowski_distance_p(x, y, p)
  else:
    result = minkowski_distance_p(x, y, p).map_inline(pow(x, 1.0 / p))

proc toTupleSeq[T: tuple](q: var HeapQueue[T]): seq[T] =
  var i = 0
  result = newSeq[T](q.len)
  while q.len > 0:
    let (val, idx) = q.pop
    result[i] = (-val, idx)
    inc i

proc toTupleSeqNorm[T: tuple](q: var HeapQueue[T], p: float): seq[T] =
  var i = 0
  result = newSeq[T](q.len)
  while q.len > 0:
    let (val, idx) = q.pop
    result[i] = (pow(-val, 1.0 / p), idx)
    inc i

proc query[T](tree: KDTree[T],
              x: Tensor[T],
              k = 1, # number of neighbors to yield
              eps = 0.0,
              p = 2.0,
              distanceUpperBound = Inf): seq[tuple[d: T, i: int]] =
  echo tree.maxes
  echo x
  var side_distancesT = map2_inline(x .- tree.maxes,
                                    tree.mins .- x):
    max(0, max(x, y))
  var min_distance: T
  var distance_upper_bound = distance_upper_bound
  if classify(p) != fcInf:
    side_distancesT = side_distancesT.map_inline(pow(x, p))
    min_distance = sum(side_distancesT)
  else:
    min_distance = max(side_distancesT) # , axis = 0)

  var side_distances = side_distancesT.toRawSeq
  # priority queue for chasing nodes
  # - min distance between cell and target
  # - distance between nearest side of cell and target
  # - head node of cell
  var q = initHeapQueue[(T, seq[T], Node[T])]()
  q.push (min_distance, side_distances, tree.tree)

  # priority queue for nearest neighbors
  # - (- distance ** p)
  # - index
  var neighbors = initHeapQueue[(T, int)]()

  var epsfac: T
  if eps == 0.T:
    epsfac = 1.T
  elif classify(p) == fcInf:
    epsfac = T(1 / (1 + eps))
  else:
    epsfac = T(1 / pow(1 + eps, p))

  if classify(p) != fcInf and classify(distance_upper_bound) != fcInf:
    distance_upper_bound = pow(distance_upper_bound, p)

  var node: Node[T]
  var data: Tensor[T]
  while q.len > 0:
    (min_distance, side_distances, node) = pop q
    case node.kind
    of tnLeaf:
      # brute force for remaining
      data = tree.data.fancyIndex(node.idx)
      let ds = minkowski_distance_p(data, x.unsqueeze(axis = 0), p).squeeze
      for i in 0 ..< ds.size:
        if ds[i] < distance_upper_bound:
          if neighbors.len == k:
            discard pop(neighbors)
          neighbors.push( (-ds[i], node.idx[i]) )
          if neighbors.len == k:
            distance_upper_bound = -neighbors[0][0]
    of tnInner:
      if min_distance > distance_upper_bound * epsfac:
        # nearest cell, done, bail out
        break
      # compute min distance to children, push them
      var near: Node[T]
      var far: Node[T]
      if x[node.split_dim] < node.split:
        (near, far) = (node.lesser, node.greater)
      else:
        (near, far) = (node.greater, node.lesser)
      q.push( (min_distance, side_distances, near) )

      var sd = side_distances
      if classify(p) == fcInf:
        min_distance = max(min_distance, abs(node.split - x[node.split_dim]))
      elif p == 1:
        sd[node.split_dim] = abs(node.split - x[node.split_dim])
        min_distance = min_distance - side_distances[node.split_dim] + sd[node.split_dim]
      else:
        sd[node.split_dim] = pow(abs(node.split - x[node.split_dim]), p)
        min_distance = min_distance - side_distances[node.split_dim] + sd[node.split_dim]

      if min_distance <= distance_upper_bound * epsfac:
        q.push( (min_distance, sd, far) )

  if classify(p) == fcInf:
    result = toTupleSeq neighbors
  else:
    result = toTupleSeqNorm(neighbors, p)

when isMainModule:
  import arraymancer

  let xs = randomTensor(1000, 1.0)
  let ys = randomTensor(1000, 1.0)
  let t = stack([xs, ys]).transpose

  let kd = kdTree(t)
  #echo kd.tree.repr

  let ps = randomTensor(2, 1.0)
  echo ps.shape
  echo ps
  let idxs = kd.query(ps, k = 3)

  for x in idxs:
    echo x
    echo t[x[1], _]