# Copyright (c) 2018 Mamy André-Ratsimbazafy and the Arraymancer contributors
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.
import math, random, tables

import
  ../../tensor/tensor, ../../linear_algebra/linear_algebra

proc euclidean_distance [T: SomeReal](u: Tensor[T], v: Tensor[T], squared: bool = false): T =
  ## Calculates the euclidean distance
  ## Inputs:
  ##  - u: a tensor
  ##  - v: a tensor
  ##  - squared: whether or not to square the result
  ## sqrt( (u - v) dot (u - v) )
  ## which can be reformulated
  ## sqrt((u dot u) - 2 * (u dot v) + (v dot v))
  var u_v = (u .- v).reshape(u.shape[1])
  result = dot(u_v, u_v)
  if not squared:
    result = sqrt(result) 

proc cumsum [T: SomeReal](p: Tensor[T]): Tensor[T] =
  ## Calculates the cumulative sum of a vector.
  ## Inputs:
  ##  - p: the tensor to cumulatively sum
  assert p.rank == 1
  assert p.shape[1] == 0
  var n_rows = p.shape[0]
  result = p
  for i in 1..<n_rows:
    result[i] += result[i-1]

proc init_random [T: SomeReal](x: Tensor[T], n_clusters: int): Tensor[T] =
  ## Helper method to randomly assign the initial centroids
  result = newTensor[T](n_clusters, x.shape[1])
  for i in 0..<n_clusters:
    var random_point = rand(x.shape[0])
    result[i, _] = x[random_point, _]

proc get_closest_centroid [T: SomeReal](x: Tensor[T], centroids: Tensor[T], cid: int): int =
  ## Helper method to get the closest centroid
  var
    closest_dist = Inf
    closest_centroid_id: int
  for closest in 0..<cid:
    if euclidean_distance(x[cid, _], centroids[closest, _], squared = true) < closest_dist:
      closest_centroid_id = closest
  return closest_centroid_id

proc get_candidates [T: SomeReal](n: int, distances: Tensor[T]): Tensor[int] =
  ## Sample candidates with probability weighted by the distances
  var
    probs = cumsum(distances ./ @[distances.sum].toTensor)
    candidates = newTensor[int](n)
  for t in 0..<n:
    block sampling:
      for i in 0..<probs.shape[0]:
        if rand(1.0) <= probs[i]:
          candidates[t] = i
          break sampling
  return candidates

proc get_distances [T: SomeReal](point: int, x: Tensor[T], squared: bool = false): Tensor[T] =
  ## Helper method to get distances from one point to all other points in a matrix.
  var 
    n_rows = x.shape[0]
    dists = newTensor[T](n_rows)
  for i in 0..<n_rows:
    dists[i] = euclidean_distance(x[point, _], x[i, _], squared)
  return dists

proc init_plus_plus [T: SomeReal](x: Tensor[T], n_clusters: int): Tensor[T] = 
  ## Helper method to use the KMeans++ heuristic for initial centroids
  ##  - x: a tensor of input data with rank of 2
  ##  - n_clusters: the number of centroids to initialize 
  ##
  ## k-means++ is an initialization heuristic for the standard k-means algorithm.
  ## It uses a simple randomized seeding technique that is O(log k) competive with
  ## the optimal clustering and speeds up convergance. The implementation here is based 
  ## loosely ## off the one in scikit-learn: 
  ## https://github.com/scikit-learn/scikit-learn/blob/a24c8b46/sklearn/cluster/k_means_.py#L43
  ##
  ## The intuition behind the algorithm is to choose centroids that are far away from each other,
  ## by sampling each centroid weighted by the distance function. 
  ##
  ## See Arthur and Vassilvitskii (2007) "k-means++: The Advantages of Careful Seeding"
  assert x.rank == 2
  let
    n_rows = x.shape[0]
    n_cols = x.shape[1]
    first_centroid = rand(n_rows)
    n_trials = int(2 + (ln(float(n_clusters))))
  var centroids = newTensor[T](n_clusters, n_cols)

  # Assign first centroid
  centroids[0, _] = x[first_centroid, _]

  for cid in 1..<n_clusters:
    var 
      closest_centroid_id = get_closest_centroid(x, centroids, cid)
      distances = get_distances(closest_centroid_id, x, squared = true)
      candidates = get_candidates(n_trials, distances)
      best_candidate_inertia: T
      best_candidate_id: int

    # Find best candidate by highest inertia
    for candidate_idx in 0..<n_trials:          
      var
        candidate_id = candidates[candidate_idx]
        candidate_inertia = get_distances(candidate_id, x, squared = true).sum
      if candidate_inertia >= best_candidate_inertia:
        best_candidate_inertia = candidate_inertia
        best_candidate_id = candidate_id

    # Assign best candidate for current centroid
    centroids[cid, _] = x[best_candidate_id, _]

  return centroids

proc assign_labels [T: SomeReal](x: Tensor[T], n_clusters = 10, tol: float = 0.0001, max_iters = 300, random = false):
  tuple[labels: Tensor[int], centroids: Tensor[T], inertia: T] =
  ## K-Means Clustering label assignment
  ##  - x: A matrix of shape [Nb of observations, Nb of features]
  ##  - n_clusters: The number of cluster centroids to compute
  ##  - tol: early stopping criterion if centroids move less than this amount on an iteration
  ##  - max_iters: maximum total passes over x before stopping
  ##  - random: whether or not to start the centroid coordinates randomly. By default, uses kmeans++ to choose the centroids.
  ##
  ## Returns:
  ##  - labels: cluster labels produced by the algorithm - a tensor of shape [Nb of observations, 1]
  ##  - centroids: the coordinates of the centroids - a matrix of shape [n_clusters, x.shape[1]]
  ##  - inertia: a measurement of distortion across all clustering labeling - a single value of type T
  let 
    n_rows = x.shape[0]
    n_cols = x.shape[1]
  assert x.rank == 2
  assert n_clusters <= n_cols

  var 
    iters: int
    inertia: T
    previous_inertia: T 
    labels = newTensor[int](n_rows)
    centroids = newTensor[T](n_clusters, n_cols)
    # Keep a running total of the count and total
    # to calculate the means. Keyed by centroid_id
    counts = initTable[int, int]()
    totals = initTable[int, Tensor[T]]()

  if not random:
    centroids = init_plus_plus(x, n_clusters)
  else:
    centroids = init_random(x, n_clusters)
  
  # Populate defaults in tables
  for i in 0..<n_clusters:
    counts[i] = 0
    totals[i] = newTensor[T](n_cols)

  # Assign labels
  block update_centroids:
    while true:
      iters += 1
      # Store previous inertia
      previous_inertia = inertia
      inertia = 0.0

      for row_idx in 0..<n_rows:
        var 
          min_dist: T = Inf
          min_label: int 
        for centroid_idx in 0..<n_clusters:
          var dist = euclidean_distance(x[row_idx, _], centroids[centroid_idx, _])
          if min_dist == -1 or dist < min_dist:
            min_dist = dist
            min_label = centroid_idx

        # Update inertia
        inertia += min_dist
        # Assign that cluster id the labels tensor
        labels[row_idx] = min_label

        # Update the counts
        counts[min_label] += 1
        # Update the running total
        totals[min_label] += x[row_idx, _]
      
      # Stopping criteria
      if (inertia - previous_inertia) <= tol or (iters >= max_iters):
        break update_centroids
      
      # Update centroids, update inertia, if points have been assigned to them
      for i in 0..<n_clusters:
        # Avoid NaNs
        if counts[i] > 0:
          var count = @[counts[i]].toTensor.astype(T)
          centroids[i, _] = (totals[i] ./ count).reshape(1, n_cols)

  return (labels: labels, centroids: centroids, inertia: inertia)

proc kmeans*[T: SomeReal](x: Tensor[T], n_clusters = 10, tol: float = 0.0001, n_init = 10, max_iters = 300, seed = 1000, random = true):
  tuple[labels: Tensor[T], centroids: Tensor[T], inertia: T] {.noInit.} =
  ## K-Means Clustering
  ## Inputs:
  ##  - x: A matrix of shape [Nb of observations, Nb of features]
  ##  - n_clusters: The number of cluster centroids to compute
  ##  - tol: early stopping criterion if centroids move less than this amount on an iteration
  ##  - max_iters: maximum total passes over x before stopping
  ##  - seed: random seed for reproducability
  ##
  ## Returns:
  ##  - a tuple of:
  ##    - Cluster labels : a matrix of shape [Nb of observations, 1]
  ##    - Centroid coordinates : a matrix of shape [n_clusters, Nb of features]
  ##    - Inertia: the sum of sq distances from each point to its centroid 
  let
    n_rows = x.shape[0]
    n_cols = x.shape[1]
  assert x.rank == 2
  assert n_clusters <= n_cols
  var 
    inertias = newTensor[T](n_init)
    labels = newSeq[Tensor[T]](n_init)
    centroids = newSeq[Tensor[T]](n_init)

  randomize(seed)

  for i in 0..<n_init:
    var output = x.assign_labels(n_clusters, tol, max_iters)
    labels[i] = output.labels.astype(T)
    inertias[i] = output.inertia
    centroids[i] = output.centroids

  let i = inertias.find(inertias.min)
  return (labels[i], centroids[i], inertias[i]) 

proc kmeans*[T: SomeReal](x: Tensor[T], centroids: Tensor[T]): Tensor[T] {.noInit.} =
  ## K-Means Clustering
  ## Inputs:
  ##  - x: A matrix of shape [Nb of observations, Nb of features]
  ##  - centroids: A matrix of shape [Nb of centroids, Nb of features]
  ##
  ## Returns:
  ##  - Cluster labels : a matrix of shape [Nb of observations, 1]
  let 
    n_rows = x.shape[0]
    n_cols = x.shape[1]
    n_clusters = centroids.shape[0]
  assert x.rank == 2
  assert centroids.rank == 2
  assert n_clusters <= n_cols

  var labels = newTensor[int](n_rows)

  for row_idx in 0..<n_rows:
    var 
      min_dist: T = -1.0
      min_label = -1
    for centroid_idx in 0..<n_clusters:
      var dist = euclidean_distance(x[row_idx, _], centroids[centroid_idx, _])
      if min_dist == -1 or dist < min_dist:
        min_dist = dist
        min_label = centroid_idx
    
    labels[row_idx] = min_label 
  
  return labels.astype(T)