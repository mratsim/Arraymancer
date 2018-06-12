# Copyright (c) 2018 Mamy André-Ratsimbazafy and the Arraymancer contributors
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

import
  ../../tensor/tensor, ../../linear_algebra/linear_algebra
import math, random, tables

proc euclidean_distance [T: SomeFloat](u: Tensor[T], v: Tensor[T]): T =
  ## Helper method to calculate the euclidean distance
  ## sqrt( (u - v) dot (u - v) )
  var u_v = (u .- v).reshape(u.shape[1])
  return sqrt(dot(u_v, u_v))

proc init_random [T: SomeFloat](x: Tensor[T], n_clusters: int): Tensor[T] =
  ## Helper method to randomly assign the initial centroids
  var centroids = newTensor[T](n_clusters, x.shape[1])
  # Produce a random coordinate for each component
  for i in 0..<n_clusters:
    var random_point = rand(x.shape[1])
    centroids[i, _] = x[random_point, _]
  return centroids

proc init_plus_plus [T: SomeFloat](x: Tensor[T], n_clusters: int): Tensor[T] = 
  ## Helper method to use the KMeans++ heuristic for initial centroidsk
  var
    potential: T
    n_rows = x.shape[0]
    n_cols = x.shape[1]
    # Choose first centroid randomly
    center_id = rand(n_rows)
    # Container for centroids
    centroids = newTensor[T](n_clusters, n_cols)

  for c in 0..<n_clusters:
    var
      probs = newTensor[T](n_rows)
      random_values = newTensor[T](n_rows)
      distances = newTensor[T](n_rows)
      candidate_probability = rand(1.0)
      candidate_id = -1

    # Get distances
    for i in 0..<n_rows:
      distances[i] = euclidean_distance(x[center_id, _], x[i, _])
    
    # Probabilities, weighted by distance
    var total_distance = distances.sum
    if total_distance > 0:
      probs = distances ./ @[total_distance].toTensor

    for i in 1..<n_rows:
      # Cum sum
      probs[i] += probs[i-1]
      if candidate_probability < probs[i]:
        # If unchosen, or further than before
        if candidate_id == -1 or distances[candidate_id] < distances[i]:
          # Assign as new centroid
          centroids[c, _] = x[i, _]
  return centroids


proc assign_labels [T: SomeFloat](x: Tensor[T], n_clusters = 10, tol: float = 0.0001, max_iters = 300, random: bool = false): tuple[labels: Tensor[int], centroids: Tensor[T], inertia: T] =
  ## K-Means Clustering label assignment

  ##   - x: A matrix of shape [Nb of observations, Nb of features]
  ##   - n_clusters: The number of cluster centroids to compute
  ##   - tol: early stopping criterion if centroids move less than this amount on an iteration
  ##   - max_iters: maximum total passes over x before stopping
  ##   - random: whether or not to start the centroid coordinates randomly. By default, uses kmeans++ to choose the centroids.
  ##
  ## Returns:
  ##   - labels: cluster labels produced by the algorithm - a tensor of shape [Nb of observations, 1]
  ##   - centroids: the coordinates of the centroids - a matrix of shape [n_clusters, x.shape[1]]
  ##   - inertia: a measurement of distortion across all clustering labeling - a single value of type T
  let 
    n_rows = x.shape[0]
    n_cols = x.shape[1]
  assert x.rank == 2
  assert n_clusters <= n_cols

  var 
    iters: int = 0
    inertia: T
    previous_inertia: T 
    labels = newTensor[int](n_rows)
    centroids = newTensor[T](n_clusters, n_cols)

  if not random:
    centroids = init_plus_plus(x, n_clusters)
  else:
    centroids = init_random(x, n_clusters)


  # Keep a running total of the count and total
  # to calculate the means. Keyed by centroid_id
  var 
    counts = initTable[int, int]()
    totals = initTable[int, Tensor[T]]()
  
  # Populate defaults in tables
  for i in 0..<n_clusters:
    counts[i] = 0
    totals[i] = newTensor[T](n_cols)

  # Assign labels
  block update_centroids:
    while true:
      iters += 1

      # Store previous inertia and reset
      previous_inertia = inertia
      var inertia: T

      for row_idx in 0..<n_rows:
        var 
          min_dist: T = -1.0
          min_label = -1
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

proc kmeans*[T: SomeFloat](x: Tensor[T], n_clusters = 10, tol: float = 0.0001, n_init = 10, max_iters = 300, seed = 1000, transform = true): Tensor[T] =
  ## K-Means Clustering
  ## Inputs:
  ##   - x: A matrix of shape [Nb of observations, Nb of features]
  ##   - n_clusters: The number of cluster centroids to compute
  ##   - tol: early stopping criterion if centroids move less than this amount on an iteration
  ##   - max_iters: maximum total passes over x before stopping
  ##   - seed: random seed for reproducability
  ##   - transform: whether or not to return the cluster labels or centroids
  ##
  ## Returns:
  ##   - Cluster labels if transform is true: a matrix of shape [Nb of observations, 1]
  ##   - Centroid coordinates if transform is false: a matrix of shape [n_clusters, Nb of features]
  ##    
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

  let best_clustering = inertias.find(inertias.min)
  if transform:
    return labels[best_clustering]
  else:
    return centroids[best_clustering]

proc kmeans*[T: SomeFloat](x: Tensor[T], centroids: Tensor[T]): Tensor[T] =
  ## K-Means Clustering
  ## Inputs:
  ##   - x: A matrix of shape [Nb of observations, Nb of features]
  ##   - centroids: A matrix of shape [Nb of centroids, Nb of features]
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