import ../../src/arraymancer
import unittest, random, sequtils, algorithm

proc `=~=`(x, y: float, eps = 1e-6): bool =
  ## simple float compare with epsilon to not have to worry about perfect scipy / nim
  ## comparisons
  result = abs(x - y) < eps

template `=~=`(x, y: Tensor[float], eps = 1e-6): bool =
  ## simple float compare with epsilon to not have to worry about perfect scipy / nim
  ## comparisons
  check x.size == y.size
  result = true
  for i in 0 ..< x.size:
    result = result and x[i] =~= y[i]

template compare(nimResTup, expDists, expIdxs: untyped): untyped =
  # sort the output by the index (arg 1)
  let nimRes = zip(nimResTup[0].toRawSeq,
                   nimResTup[1].toRawSeq).sortedByIt(it[1])
  let expRes = zip(expDists, expIdxs).sortedByIt(it[1])
  check nimRes.len == expRes.len
  for i in 0 ..< nimRes.len:
    check nimRes[i][0] =~= expRes[i][0]
    check nimRes[i][1] == expRes[i][1]

when true:
  # for expected numbers
  import nimpy

suite "k-d tree: k = 2":
  # choose fixed randomization
  randomize(42)
  # create 1000 random x/y points and combine to (1000, 2) tensor
  let xs = randomTensor(1000, 1.0)
  let ys = randomTensor(1000, 1.0)
  let t = stack([xs, ys]).transpose

  # create a tree with 16 leaves, balanced (the default)
  let kd = kdTree(t, leafSize = 16, balancedTree = true)

  # create a random point to sample at
  let ps = randomTensor(2, 1.0)

  test "General k-d tree checks":
    check kd.size == 127

  test "Query for 3 points around a point":

    # query for 3 points around `ps`
    let nimResTup = kd.query(ps, k = 3)
    let expDists = @[0.021058219901320622, 0.021894545141264914, 0.0327512983314587]
    let expIdxs = @[554, 116, 824]
    compare(nimResTup, expDists, expIdxs)

    ## The expected results for this part were obtained via scipy using:
    when false:
      let scipy = pyImport("scipy.spatial")
      let np = pyImport("numpy")
      # `xs` and `ys` are the above of course
      let tree = scipy.cKDTree(np.array([xs.toRawSeq, ys.toRawSeq]).T)
      # scipy returns array of dist, array of ids. Should do the same I guess
      # Having a seq[tuple] as return is bad.
      let scipyResPy = tree.query(ps.toRawSeq, 3)
      let scipyDists = scipyResPy[0].mapIt(it.to(float))
      let scipyIdxs = scipyResPy[1].mapIt(it.to(int))
      echo "--------"
      echo scipyDists
      echo scipyIdxs
      echo "--------"

  test "Query for all points around point in radius":

    let nimResTup = kd.query_ball_point(ps, 0.1)
    let expDists = @[0.08524257858478503, 0.09281827132268332, 0.07985522103061606, 0.09754373439392204,
                     0.09279351538329877, 0.08596229531983582, 0.09266667737463005, 0.08836005446023058,
                     0.07192545436693633, 0.06733743101154321, 0.048446401077739745, 0.04440091977837231,
                     0.08062012422931063, 0.0327512983314587, 0.07662877346256429, 0.021894545141264914,
                     0.06521004969453695, 0.021058219901320622, 0.052374047551014394, 0.07778513512726905,
                     0.07135584073961694, 0.0732687610356244, 0.052981060897594655, 0.08459285997107174]
    let expIdxs = @[252, 124, 222, 946, 775, 504, 41, 395, 587, 965, 65, 750, 771, 824, 976, 116, 16, 554, 85, 236, 827,
                    648, 320, 964]
    compare(nimResTup, expDists, expIdxs)

    ## The expected results for this part were obtained via scipy using:
    when false:
      let scipy = pyImport("scipy.spatial")
      let np = pyImport("numpy")
      # `xs` and `ys` are the above of course
      let tree = scipy.cKDTree(np.array([xs.toRawSeq, ys.toRawSeq]).T)
      # query_ball_point only returns indices in scipy. Compute distances manually for each point
      let pdt = tree.query_ball_point(ps.toRawSeq, 0.1)
      var dists = newSeq[float]()
      for i in pdt:
        let idx = i.to(int)
        dists.add minkowski_distance(t[idx, _].squeeze, ps)[0]
      echo dists
      echo pdt

suite "k-d tree: k = 5":
  # create individual tensors to feed them to scipy
  let x0 = randomTensor(1000, 1.0)
  let x1 = randomTensor(1000, 1.0)
  let x2 = randomTensor(1000, 1.0)
  let x3 = randomTensor(1000, 1.0)
  let x4 = randomTensor(1000, 1.0)
  let t = stack([x0, x1, x2, x3, x4]).transpose

  # create a tree with 16 leaves, balanced (the default)
  let kd = kdTree(t, leafSize = 16, balancedTree = true)

  # create a random point to sample at
  let ps = randomTensor(5, 1.0)

  test "General k-d tree checks":
    check kd.size == 127

  test "Query for 3 points around a point":

    # query for 3 points around `ps`
    let nimResTup = kd.query(ps, k = 10)
    let expDists = @[0.18597568396567568, 0.18893577904366327, 0.2091273970672349, 0.23902028522275312,
                     0.2873669897423777, 0.3032755376678265, 0.3197172215031703, 0.3273331379135004,
                     0.3289472604965053, 0.3346825690794886]
    let expIdxs = @[374, 245, 389, 53, 784, 342, 344, 311, 182, 171]
    compare(nimResTup, expDists, expIdxs)

    ## The expected results for this part were obtained via scipy using:
    when false:
      let scipy = pyImport("scipy.spatial")
      let np = pyImport("numpy")
      # `xs` and `ys` are the above of course
      let tree = scipy.cKDTree(np.array([x0.toRawSeq, x1.toRawSeq, x2.toRawSeq, x3.toRawSeq, x4.toRawSeq]).T)
      # scipy returns array of dist, array of ids. Should do the same I guess
      # Having a seq[tuple] as return is bad.
      let scipyResPy = tree.query(ps.toRawSeq, 10)
      let scipyDists = scipyResPy[0].mapIt(it.to(float))
      let scipyIdxs = scipyResPy[1].mapIt(it.to(int))
      echo "--------"
      echo scipyDists
      echo scipyIdxs
      echo "--------"

  test "Query for all points around point in radius":
    let nimResTup = kd.query_ball_point(ps, 0.5)
    let expDists = @[0.45399285464862815, 0.47063267948837434, 0.4488851243955927, 0.4215683458202343,
                     0.42869998162302436, 0.47186440912565775, 0.39755345606859466, 0.4818053098853787,
                     0.43254771900897, 0.3197172215031703, 0.3289472604965053, 0.23902028522275312, 0.35464380388877753,
                     0.3877762051571806, 0.4551855189829431, 0.41989884394270544, 0.47213780573514846,
                     0.4631682739618477, 0.3390257492670617, 0.43180863582338097, 0.47554299732595373,
                     0.4962507460181432, 0.4720817121818903, 0.43972693362771925, 0.48273979458094113,
                     0.3658935401477406, 0.4998328526039948, 0.3764103479495621, 0.48956893885533304,
                     0.35440103301251147, 0.35720950805316515, 0.18893577904366327, 0.4913692850735096,
                     0.18597568396567568, 0.3346825690794886, 0.4893271335167886, 0.34438352514624304,
                     0.42193189802408065, 0.4573275254957121, 0.3562925931286955, 0.37265056096682925,
                     0.4966081833151957, 0.2873669897423777, 0.39598295108919646, 0.3892190105700986,
                     0.3032755376678265, 0.3273331379135003, 0.2091273970672349, 0.4752073689969398, 0.4424121195429838,
                     0.40388141135568445, 0.3610455904180944, 0.4619081460147357, 0.456243918402455, 0.493376789885083,
                     0.4437841111469847, 0.48061986212203933, 0.4987975999635133, 0.43051806670507103,
                     0.4735737911114841]
    let expIdxs = @[429, 482, 783, 590, 717, 84, 471, 650, 944, 344, 182, 53, 668, 102, 810, 85, 719, 495, 403, 924, 19,
                    395, 524, 286, 216, 865, 721, 11, 264, 597, 21, 245, 481, 374, 171, 200, 225, 596, 298, 518, 638,
                    288, 784, 351, 887, 342, 311, 389, 287, 367, 836, 217, 382, 335, 488, 989, 300, 236, 384, 173]
    compare(nimResTup, expDists, expIdxs)

    ## The expected results for this part were obtained via scipy using:
    when false:
      let scipy = pyImport("scipy.spatial")
      let np = pyImport("numpy")
      # `xs` and `ys` are the above of course
      let tree = scipy.cKDTree(np.array([x0.toRawSeq, x1.toRawSeq, x2.toRawSeq, x3.toRawSeq, x4.toRawSeq]).T)
      # query_ball_point only returns indices in scipy. Compute distances manually for each point
      let pdt = tree.query_ball_point(ps.toRawSeq, 0.5)
      var dists = newSeq[float]()
      for i in pdt:
        let idx = i.to(int)
        dists.add minkowski_distance(t[idx, _].squeeze, ps)[0]
      echo dists
      echo pdt
