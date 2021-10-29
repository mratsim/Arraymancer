# Copyright (c) 2018 Mamy André-Ratsimbazafy and the Arraymancer contributors
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

import ../../src/arraymancer, ../testutils
import
  unittest,
  math,
  random,
  sequtils

proc generateRandomClusters(): (Tensor[float], seq[int]) =
  ## Generate 3 random clusters around 3 different centers with
  ## different radii. Also returns the correct labels for each point.
  randomize(42)
  var xs = newSeq[float]()
  var ys = newSeq[float]()
  let centers = [0.1, 0.32, 0.7]
  let radii = [0.15,  0.1, 0.2]
  var labels = newSeq[int]()
  for i in 0 ..< 1000:
    let centerIdx = rand(2)
    labels.add centerIdx
    let radius = rand(radii[centerIdx])
    let angle = rand(2 * PI)
    xs.add centers[centerIdx] + radius * cos(angle)
    ys.add centers[centerIdx] + radius * sin(angle)
  result = (stack([xs.toTensor, ys.toTensor]).transpose,
            labels)

when false:
  # download broken cluster data from gist.github.com (need to compile with `-d:ssl`)
  const url = "https://gist.githubusercontent.com/Vindaar/27101232eb033462aabceb8883ad8176/raw/7e214baffbb2ee3b8b89dd16c975bd35ac5f50ca/cluster_breaking_dbscan.csv"
  import httpclient
  var client = newHttpClient()
  const outpath = "./tests/ml/data/input/broken_cluster.csv"
  writeFile(outpath, client.getContent(url))

proc main() =
  suite "[ML] Clustering":
    # Fishers Iris dataset - sans species column
    # See R. A. Fisher (1936) "The use of multiple measurements in taxonomic problems"
    let data = read_npy[float]("./tests/ml/data/input/iris_no_species.npy")

    # Keep copy of data to check for modification
    let cleanData = data.clone()

    let (randomClusters, randomLabels) = generateRandomClusters()

    test "KMeans - KMeans++ Centroids":

      var
        (labels, centroids, inertia) = data.kmeans(3)
        transformed = data.kmeans(centroids)

      let plus_plus_path = "./tests/ml/data/expected/kmeans++_output.npy"
      # To update, just uncomment this line:
      # transformed.write_npy(plus_plus_path)
      var expected = read_npy[int](plus_plus_path)

      check:
        labels == transformed
        labels == expected
        data == cleanData
        round(inertia, 4) == 99.7369

    test "Kmeans - Random Centroids":
      var
        (labels, centroids, inertia) = data.kmeans(n_clusters=3, random=true)
        transformed = data.kmeans(centroids)

      let random_path = "./tests/ml/data/expected/kmeans_random_output.npy"
      # To update, just uncomment this line:
      # transformed.write_npy(random_path)
      var expected = read_npy[int](random_path)

      check:
        labels == transformed
        labels == expected
        data == cleanData
        round(inertia, 4) == 97.0019

    test "Kmeans - More random clusters":
      var
        (labels, centroids, inertia) = randomClusters.clone.kmeans(n_clusters=3, random=false)
      # NOTE: the labels as they are assigned to the clusters are different than the random
      # labels we start with (but the clusters themselves match). Thus we will have 2 arrays
      # one for each set of labels that we fill with the correct indices (counted from 0).
      # Then we can use the labels to get the correct "index" and compare if they match.
      var lLabs = [-1, -1, -1]
      var rLabs = [-1, -1, -1]
      var curIdx = 0
      for (l, r) in zip(labels.toRawSeq, randomLabels):
        if lLabs[l] == -1:
          lLabs[l] = curIdx
          doAssert rLabs[r] == -1
          rLabs[r] = curIdx
          inc curIdx
        check lLabs[l] == rLabs[r]

    test "DBSCAN - Centroids":
      let res = randomClusters.dbscan(eps = 0.05, minSamples = 10, Euclidean)

      check res.deduplicate.len == 4 # we have a few pixels not within a cluster
      # to get invalid id `notInClusters` array:
      when false:
        for i, r in res:
          if r == -1:
            echo i
      check res.len == randomLabels.len
      let notInCluster = [200, 246, 631, 696]
      for i in 0 ..< res.len:
        if i in notInCluster:
          check res[i] == -1
        else:
          check res[i] == randomLabels[i]

    test "DBSCAN - Cluster causing out of bounds access":
      # prior to #521 this cluster caused out of bounds access during k-d tree construction
      let path = "./tests/ml/data/input/broken_cluster"
      when false: # if rerunning the download `when false` block from the top
        let t = readCsv[int](path & ".csv", skipHeader = true)
        t.writeNpy(path & ".npy")
      let expPath = "./tests/ml/data/expected/broken_cluster_expected_clustering.npy"
      let tExp = readNpy[int](expPath)
      let tRes = readNpy[int](path & ".npy")
      when not defined(release):
        echo "HINT: This test may take a while as we're clustering ~11,000 points"
      check dbscan(tRes.asType(float), eps = 30.0, minSamples = 3).toTensor == tExp


main()
GC_fullCollect()
