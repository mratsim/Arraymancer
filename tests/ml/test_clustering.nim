# Copyright (c) 2018 Mamy André-Ratsimbazafy and the Arraymancer contributors
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

import ../../src/arraymancer
import
  unittest,
  math

proc main() =
  suite "[ML] Clustering":
    # Fishers Iris dataset - sans species column
    # See R. A. Fisher (1936) "The use of multiple measurements in taxonomic problems"
    let data = read_npy[float]("./tests/ml/data/input/iris_no_species.npy")

    # Keep copy of data to check for modification
    let cleanData = data.clone()

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


main()
GC_fullCollect()
