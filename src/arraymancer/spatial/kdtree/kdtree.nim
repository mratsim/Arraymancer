#import ../tensor/tensor
import arraymancer
import sequtils, math, heapqueue, sugar, algorithm, typetraits

type
  TreeNodeKind = enum
    tnLeaf, tnInner

  Node[T] = ref object
    #level: int
    id: int
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
  ## I assume the idea is that, since the tree is built from the root node that the nodes
  ## later nodes will end up at later positions in memory. Why not just add an id field
  ## to the node, which is a unique identifier? This here is ugly, unsafe and I'm not
  ## convinced it will work under all circumstances.
  result = n1.id < n2.id #cast[int](n1.unsafeAddr) < cast[int](n2.unsafeAddr)

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

proc allEqual[T](t: Tensor[T], val: T): bool =
  ## checks if all elements of `t` are `val`
  result = true
  for x in t:
    if x != val:
      return false

proc build[T](tree: KDTree[T],
              idx: Tensor[int],
              nodeId: var int,
              #startIdx, endIdx: int
              maxes, mins: Tensor[T]): Node[T] =
              #useMedian: static bool,
              #createCompact: static bool) =
  ## recursively build the KD tree
  ##
  ## `startId` is the current ID the latest node was built with.
  if idx.size <= tree.leafSize:
    inc nodeId
    result = Node[T](id: nodeId,
                     kind: tnLeaf,
                     idx: idx,
                     children: idx.size)

  else:
    var data = tree.data[idx]
    let d = argmax((maxes .- mins).squeeze, axis = 0)[0]
    let maxVal = maxes[d]
    let minVal = mins[d]
    if maxVal == minVal:
      inc nodeId
      return Node[T](id: nodeId,
                     kind: tnLeaf,
                     idx: idx,
                     children: idx.size)
    data = squeeze(data[_, d])

    # sliding midpoint rule
    var split = (maxVal + minVal) / 2.0
    # we (ab)use nonzero to get the indices for the mask along the
    # first axis
    var lessIdx = nonzero(data <=. split)[0, _]
    var greaterIdx = nonzero(data >. split)[0, _]
    if lessIdx.size == 0:
      split = min(data)
      lessIdx = nonzero(data <=. split)[0, _]
      greaterIdx = nonzero(data >. split)[0, _]
    if greaterIdx.size == 0:
      split = max(data)
      lessIdx = nonzero(data <. split)[0, _]
      greaterIdx = nonzero(data >=. split)[0, _]
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

    inc nodeId
    let lesser = tree.build(idx[lessIdx.squeeze], nodeId, lessmaxes, mins)
    # greater starts at lesser's ID
    let greater = tree.build(idx[greaterIdx.squeeze], nodeId, maxes, greatermins)
    result = Node[T](id: nodeId,
                     kind: tnInner,
                     split_dim: d,
                     split: split,
                     lesser: lesser,
                     greater: greater)

proc buildKdTree[T](tree: var KDTree[T],
                    startIdx: Tensor[int],
                    useMedian: static bool,
                    createCompact: static bool) =
  var nodeId = 0
  tree.tree = tree.build(idx = startIdx,
                         nodeId = nodeId,
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

proc minkowski_distance_p[T](x, y: Tensor[T], p = 2.0): Tensor[T] =
  let ax = x.shape.len - 1
  if classify(p) == fcInf:
    result = max(abs(y .- x), axis = ax)
  elif p == 1:
    result = sum(abs(y .- x), axis = ax)
  else:
    result = sum(abs(y .- x).map_inline(pow(x, p)), axis = ax)

proc minkowski_distance[T](x, y: Tensor[T], p = 2.0): Tensor[T] =
  if classify(p) == fcInf or p == 1:
    result = minkowski_distance_p(x, y, p)
  else:
    result = minkowski_distance_p(x, y, p).map_inline(pow(x, 1.0 / p))

proc toTensorTuple[T, U](q: var HeapQueue[T],
                         retType: typedesc[U],
                         p = Inf): tuple[dist: Tensor[U],
                                         idx: Tensor[int]] =
  static: doAssert arity(T) == 2
  result[0] = newTensor[U](q.len)
  result[1] = newTensor[int](q.len)
  var i = 0
  if classify(p) == fcInf:
    while q.len > 0:
      let (val, idx) = q.pop
      result[0][i] = -val
      result[1][i] = idx
      inc i
  else:
    while q.len > 0:
      let (val, idx) = q.pop
      result[0][i] = pow(-val, 1.0 / p)
      result[1][i] = idx
      inc i

proc query[T](tree: KDTree[T],
              x: Tensor[T],
              k = 1, # number of neighbors to yield
              eps = 0.0,
              p = 2.0,
              distanceUpperBound = Inf): tuple[dist: Tensor[T],
                                               idx: Tensor[int]] =
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
      let ni = node.idx
      data = tree.data[ni]
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

  result = toTensorTuple(neighbors, retType = T, p = p)

import ggplotnim

proc extractXYId[T](n: Node[T]): (seq[Tensor[int]], seq[int]) =
  echo "n ", n.id
  echo n.split, " in dim ", n.split_dim
  case n.kind
  of tnLeaf:
    result = (@[n.idx], @[n.id])
  of tnInner:
    var (data, ids) = extractXYId(n.lesser)
    echo ids
    result[0].add data
    result[1].add ids
    (data, ids) = extractXYId(n.greater)
    result[0].add data
    result[1].add ids

proc walkTree(kd: KdTree): DataFrame =
  let (data, ids) = extractXYId(kd.tree)
  #?doAssert data[0].rank == 2, " is " & $data[0].rank
  var x = newSeq[float]()
  var y = newSeq[float]()
  var ids2 = newSeq[int]()
  for i in 0 ..< data.len:
    for j in data[i]:
      x.add kd.data[j, 0]
      y.add kd.data[j, 1]
      ids2.add ids[i]

  result = seqsToDf({ "x" : x,
                      "y" : y,
                      "id" : ids2 })

when isMainModule:
  import arraymancer, nimpy
  let xs = randomTensor(1000, 1.0)
  let ys = randomTensor(1000, 1.0)
  let t = stack([xs, ys]).transpose

  let kd = kdTree(t)
  #echo kd.tree.repr

  let ps = randomTensor(2, 1.0)
  echo ps.shape
  echo ps
  let nimResTup = kd.query(ps, k = 3)
  let nimRes = zip(nimResTup[0].toRawSeq,
                   nimResTup[1].toRawSeq).sortedByIt(it[1])

  echo "MAXES"
  echo kd.maxes
  echo kd.mins
  echo kd.size
  let df = kd.walkTree()
  echo df
  ggplot(df, aes("x", "y", color = factor("id"))) +
    geom_point() +
    ggsave("/tmp/kdtree.pdf")




  echo nimRes
  let scipy = pyImport("scipy.spatial")
  let np = pyImport("numpy")
  let tree = scipy.cKDTree(np.array([xs.toRawSeq, ys.toRawSeq]).T)
  # scipy returns array of dist, array of ids. Should do the same I guess
  # Having a seq[tuple] as return is bad.
  let scipyResPy = tree.query(ps.toRawSeq, 3)
  let scipyDists = scipyResPy[0].mapIt(it.to(float))
  let scipyIdxs = scipyResPy[1].mapIt(it.to(int))
  let scipyRes = zip(scipyDists, scipyIdxs).sortedByIt(it[1])
  echo scipyDists
  for i in 0 ..< nimRes.len:
    doAssert nimRes[i][0] == scipyRes[i][0]
    doAssert nimRes[i][1] == scipyRes[i][1]
