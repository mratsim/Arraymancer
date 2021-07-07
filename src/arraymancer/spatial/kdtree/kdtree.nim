import ../../tensor
import sequtils, math, heapqueue, algorithm, typetraits

#[

This module implements a k-d tree.

The code is mostly a port of the python based k-d tree in scipy (before scipy version
v1.6.0 there was a pure python k-d tree implementation).

The python code this is based on, can be found here:
https://github.com/scipy/scipy/blob/maintenance/1.5.x/scipy/spatial/kdtree.py

]#

type
  TreeNodeKind* = enum
    tnLeaf, tnInner

  Node*[T] = ref object
    level*: int              ## the level this node is at in the tree
    id*: int                 ## a unique id for this node
    #data: Tensor[T]         ## unused field to store values at `idx`
    case kind*: TreeNodeKind ## is it a leaf node or an inner node?
    of tnInner:
      lesser*: Node[T]       ## nodes on the "lesser" side of `split`
      greater*: Node[T]      ## nodes on the "greater" side of `split`
      split_dim*: int        ## the dimension this node splits the space at
      split*: float          ## the value at which the space is split in `split_dim`
    of tnLeaf:
      children*: int         ## number of indices stored in this node
      idx*: Tensor[int]      ## the indices of the input data stored in this node

  KDTree*[T] = ref object
    data*: Tensor[T]         ## k-d data stored in this tree
    leafSize*: int           ## maximum size of elements in a leaf node, default: 16
    k*: int                  ## dimension of a single data point
    n*: int                  ## number of data points
    maxes*: Tensor[T]        ## maximum values along each dimension of `n` data points
    mins*: Tensor[T]         ## minimum values along each dimension of `n` data points
    tree*: Node[T]           ## the root node of the tree
    size*: int               ## number of nodes in the tree

proc `<`[T](n1, n2: Node[T]): bool =
  ## Comparison of two nodes is done by comparing their `id`. The `id` tells us the
  ## order in which the node was constructed. This is sensible, as we *first* construct
  ## the *lesser* splitting.
  result = n1.id < n2.id

proc `<`[T](s1, s2: Tensor[T]): bool =
  ## just an internal comparison of two Tensors, which assumes that the order of two
  ## seqs matters.
  doAssert s1.size == s2.size
  result = false
  for i in 0 ..< s1.size:
    if s1[i] == s2[i]:
      # may still be decided, equal up to here
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
              level: int,
              maxes, mins: Tensor[T],
              useMedian: static bool): Node[T] =
  ## Recursively build the KD tree
  ##
  ## `nodeId` is the current ID the latest node was built with. It is being increased
  ## monotonically during the construction process.
  # increase the node id, as  we will return exactly one new node from here
  inc nodeId
  mixin `_`
  if idx.size <= tree.leafSize:
    result = Node[T](id: nodeId,
                     level: level,
                     kind: tnLeaf,
                     idx: idx,
                     children: idx.size)
  else:
    var data = tree.data[idx]
    # select dimension with largest difference in mins / maxes
    let d = argmax((maxes -. mins).squeeze, axis = 0)[0]
    let maxVal = maxes[d]
    let minVal = mins[d]
    if maxVal == minVal:
      return Node[T](id: nodeId,
                     level: level,
                     kind: tnLeaf,
                     idx: idx,
                     children: idx.size)
    # get all data points in this largest dimension `d`
    data = squeeze(data[_, d])

    # sliding midpoint rule
    when useMedian:
      # due to squeezing onto the "largest" dimension, `data` here is always 1D
      var split = data.percentile(50) # compute median of data
    else:
      var split = (maxVal + minVal) / 2.0 # take mean between min / max
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
        raise newException(ValueError, "Bad input data: " & $data & ". All values are the same."
        )
      split = data[0]
      lessIdx = arange(data.size - 1)
      greaterIdx = toTensor @[data.size - 1]

    var lessmaxes = maxes.clone()
    lessmaxes[d] = split
    var greatermins = mins.clone()
    greatermins[d] = split

    # Note: cannot squeeze with args in `[]`, currently broken in accessor macro
    let lesserArg = lessIdx.squeeze(axis = 0)
    let lesser = tree.build(idx[lesserArg], nodeId, level + 1, lessmaxes, mins, useMedian = useMedian)
    let greaterArg = greaterIdx.squeeze(axis = 0)
    let greater = tree.build(idx[greaterArg], nodeId, level + 1, maxes, greatermins, useMedian = useMedian)
    result = Node[T](id: nodeId,
                     level: level,
                     kind: tnInner,
                     split_dim: d,
                     split: split,
                     lesser: lesser,
                     greater: greater)

proc buildKdTree[T](tree: var KDTree[T],
                    startIdx: Tensor[int],
                    useMedian: static bool) =
  ## Starts the build process based on the root node.
  var nodeId = 0
  tree.tree = tree.build(idx = startIdx,
                         level = 1, # start level is 1 (root node is 0)
                         nodeId = nodeId,
                         maxes = tree.maxes,
                         mins = tree.mins,
                         useMedian = useMedian)
  tree.size = nodeId # last node id is the total number of nodes in the tree

proc kdTree*[T](data: Tensor[T],
                leafSize = 16,
                copyData = true,
                balancedTree: static bool = true): KDTree[T] =
  ## Builds a k-d tree based on the input `data`.
  ##
  ## `data` must be of shape `(n, k)` where `n` is the number of input data points and
  ## `k` the dimension of each point.
  ##
  ## `leafSize` is the maximum number of elements to be stored in a single leaf node.
  ##
  ## If `balancedTree` is `true`, we split along the most separated axis at the *median*
  ## point. Otherwise we split in the middle. The former leads to slightly longer
  ## build times, as we have to compute the median (which implies sorting the data along
  ## the axis), but results in a more balanced tree.
  result = new(KDTree[T])
  result.data = if copyData: data.clone() else: data
  if data.shape.len != 2:
    raise newException(ValueError, "Data must have 2 dimensions with layout `(n, k)`, where " &
      "`n` is the number of `k` dimensional points.")
  result.n = data.shape[0]
  result.k = data.shape[1]
  result.leafSize = leafSize
  doAssert leafSize > 0, "leafSize must be at least 1!"

  result.maxes = data.max(axis = 0).squeeze
  result.mins = data.min(axis = 0).squeeze

  result.buildKdTree(arange[int](result.n),
                     useMedian = balancedTree)

proc minkowski_distance_p[T](x, y: Tensor[T], p = 2.0): Tensor[T] =
  ## Implementation of the minkowski distance that computes the `p` power
  ## of the distance.
  let ax = x.shape.len - 1
  if classify(p) == fcInf:
    result = max(abs(y -. x), axis = ax)
  elif p == 1:
    result = sum(abs(y -. x), axis = ax)
  else:
    result = sum(abs(y -. x).map_inline(pow(x, p)), axis = ax)

proc minkowski_distance[T](x, y: Tensor[T], p = 2.0): Tensor[T] =
  ## Compute the minkowski distance with power `p`
  ## It is defined as:
  ##
  ## d_M = (Σ_i |x_i - y_i|^p)^(1/p)
  ##
  ## where `i` runs over all dimensions of the points `x` and `y`.
  ##
  ## which reduces to the manhatten distance for p = 1 and the euclidean
  ## distance for p = 2.
  if classify(p) == fcInf or p == 1:
    result = minkowski_distance_p(x, y, p)
  else:
    result = minkowski_distance_p(x, y, p).map_inline(pow(x, 1.0 / p))

proc toTensorTuple[T, U](q: var HeapQueue[T],
                         retType: typedesc[U],
                         p = Inf): tuple[dist: Tensor[U],
                                         idx: Tensor[int]] =
  ## Helper proc to convert the contents of the HeapQueue to a tuple of
  ## two tensors.
  ##
  ## The heap queue here is used to accumulate neighbors in the `query` proc. It
  ## stores the distance to a neighbor from the user given point and the index of
  ## that point in the input tensor.
  static: doAssert arity(T) == 2, "Generic type `T` must be a tuple of types " &
    "`(A, int)` where A is the type stored in the KD tree."
  var vals = newTensorUninit[U](q.len)
  var idxs = newTensorUninit[int](q.len)
  var i = 0
  if classify(p) == fcInf:
    while q.len > 0:
      let (val, idx) = q.pop
      vals[i] = -val
      idxs[i] = idx
      inc i
  else:
    while q.len > 0:
      let (val, idx) = q.pop
      vals[i] = pow(-val, 1.0 / p)
      idxs[i] = idx
      inc i
  result = (vals, idxs)

proc queryImpl[T](
  tree: KDTree[T],
  x: Tensor[T], # data point to query around
  k: int, # number of neighbors to yield
  radius: T, # radius to yield in. If `k` given is the upper search radius
  eps: float,
  p: float,
  yieldNumber: static bool # if true yield `k` closest neighbors, else all in `radius`
                ): tuple[dist: Tensor[T],
                         idx: Tensor[int]] =
  ## Implementation of a `query` routine for a `KDTree`. Depending on the arguments
  ## and the static `yieldNumber` arguments it returns:
  ## - the `k` neighbors around `x` within a maximum `radius` (`yieldNumber = true`)
  ## - all points around `x` within `radius` (`yieldNumber = false`)
  var side_distances = map2_inline(x -. tree.maxes,
                                    tree.mins -. x):
    max(0, max(x, y))

  var min_distance: T
  var distanceUpperBound = radius
  if classify(p) != fcInf:
    side_distances = side_distances.map_inline(pow(x, p))
    min_distance = sum(side_distances)
  else:
    min_distance = max(side_distances)

  # priority queue for chasing nodes
  # - min distance between cell and target
  # - distance between nearest side of cell and target
  # - head node of cell
  var q = initHeapQueue[(T, Tensor[T], Node[T])]()
  q.push (min_distance, side_distances.clone, tree.tree)

  # priority queue for nearest neighbors, i.e. our result
  # - (- distance ** p) from input `x` to current point
  # - index of point in `KDTree's` data
  var neighbors = initHeapQueue[(T, int)]()

  # compute a sensible epsilon
  var epsfac: T
  if eps == 0.T:
    epsfac = 1.T
  elif classify(p) == fcInf:
    epsfac = T(1 / (1 + eps))
  else:
    epsfac = T(1 / pow(1 + eps, p))

  # normalize the radius to the correct power
  if classify(p) != fcInf and classify(distanceUpperBound) != fcInf:
    distanceUpperBound = pow(distanceUpperBound, p)

  var node: Node[T]
  while q.len > 0:
    (min_distance, side_distances, node) = pop q
    case node.kind
    of tnLeaf:
      # brute force for remaining elements in leaf node
      let ni = node.idx
      let ds = minkowski_distance_p(tree.data[ni], x.unsqueeze(axis = 0), p).squeeze
      for i in 0 ..< ds.size:
        if ds[i] < distanceUpperBound:
          when yieldNumber:
            if neighbors.len == k:
              discard pop(neighbors)
          neighbors.push( (-ds[i], node.idx[i]) )
          when yieldNumber:
            if neighbors.len == k:
              distanceUpperBound = -neighbors[0][0]
    of tnInner:
      if min_distance > distanceUpperBound * epsfac:
        # nearest cell, done, bail out
        break
      # compute min distance to children, push them
      var near: Node[T]
      var far: Node[T]
      if x[node.split_dim] < node.split:
        (near, far) = (node.lesser, node.greater)
      else:
        (near, far) = (node.greater, node.lesser)
      q.push( (min_distance, side_distances.clone, near) )

      var sd = side_distances.clone # clone to avoid reference semantic issues
      if classify(p) == fcInf:
        min_distance = max(min_distance, abs(node.split - x[node.split_dim]))
      elif p == 1:
        sd[node.split_dim] = abs(node.split - x[node.split_dim])
        min_distance = min_distance - side_distances[node.split_dim] + sd[node.split_dim]
      else:
        sd[node.split_dim] = pow(abs(node.split - x[node.split_dim]), p)
        min_distance = min_distance - side_distances[node.split_dim] + sd[node.split_dim]

      if min_distance <= distanceUpperBound * epsfac:
        q.push( (min_distance, sd, far) )
  # extract all information from heap queue and return as tuple
  result = toTensorTuple(neighbors, retType = T, p = p)

proc query*[T](tree: KDTree[T],
               x: Tensor[T],
               k = 1, # number of neighbors to yield
               eps = 0.0,
               p = 2.0,
               distanceUpperBound = Inf): tuple[dist: Tensor[T],
                                                idx: Tensor[int]] =
  ## Queries the k-d `tree` at point `x` for the `k` closest neighbors.
  ##
  ## `distanceUpperBound` can be set to stop the search even if less than `k` neighbors have
  ## been found within that hyperradius.
  ##
  ## `eps` is the relative epsilon for distance comparison by which `distanceUpperBound` is scaled.
  ##
  ## `p` is the power to use in the Minkowski metric. This affects the way distances between points
  ## are computed.
  ## Typical values:
  ## - p = 1: Manhattan distance
  ## - p = 2: Euclidean distance
  result = tree.queryImpl(x = x, k = k, radius = distanceUpperBound,
                          eps = eps,
                          p = p,
                          yieldNumber = true)

proc query_ball_point*[T](tree: KDTree[T],
                          x: Tensor[T], # point to search around
                          radius: float, # hyperradius around `x`
                          eps = 0.0,
                          p = 2.0
                         ): tuple[dist: Tensor[T],
                                  idx: Tensor[int]] =
  ## Queries the k-d `tree` around point `x` for all points within the hyperradius `radius`.
  ##
  ## `eps` is the relative epsilon for distance comparison by which `distanceUpperBound` is scaled.
  ##
  ## `p` is the power to use in the Minkowski metric. This affects the way distances between points
  ## are computed and thus directly affects what a "radius" is.
  ## Typical values:
  ## - p = 1: Manhattan distance
  ## - p = 2: Euclidean distance
  # TODO: this might better be implemented using a hyperrectangle to ignore more parts of
  # the space?
  result = tree.queryImpl(x = x, k = 0, # irrelevant for `yieldNumber = false`
                          radius = radius,
                          eps = eps,
                          p = p,
                          yieldNumber = false)
