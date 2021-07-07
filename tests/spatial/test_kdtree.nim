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
    # sort the output by the index (arg 1)
    let nimRes = zip(nimResTup[0].toRawSeq,
                     nimResTup[1].toRawSeq).sortedByIt(it[1])

    let expDists = [0.017994668611127754, 0.028798502625867597, 0.019484735194430423]
    let expIdxs = [258, 294, 934]
    check nimRes.len == 3
    for i in 0 ..< 3:
      check nimRes[i][0] =~= expDists[i]
      check nimRes[i][1] == expIdxs[i]

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
    let nimRes = zip(nimResTup[0].toRawSeq,
                     nimResTup[1].toRawSeq).sortedByIt(it[1])

    let expDists = @[0.0919767, 0.0895979, 0.086663, 0.0858158, 0.0829248, 0.075204, 0.0743377,
                     0.0733601, 0.070959, 0.0634458, 0.0634204, 0.0594141, 0.0474656, 0.0442657,
                     0.0410204, 0.0409415, 0.0395273, 0.0370934, 0.0313839, 0.0292735, 0.0287985,
                     0.0194847, 0.0179947]
    let expIdxs = @[793, 522, 723, 600, 962, 533, 429, 919, 872, 684, 895, 929, 681, 404, 362, 758,
                    97, 195, 529, 682, 294, 934, 258]
    let expRes = zip(expDists, expIdxs).sortedByIt(it[1])
    check nimRes.len == expRes.len
    for i in 0 ..< nimRes.len:
      check nimRes[i][0] =~= expRes[i][0]
      check nimRes[i][1] == expRes[i][1]

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
    # sort the output by the index (arg 1)
    let nimRes = zip(nimResTup[0].toRawSeq,
                     nimResTup[1].toRawSeq).sortedByIt(it[1])

    let expDists = @[0.14667821419698868, 0.19483515651264544, 0.2260793690650405,
                     0.22933987791539506, 0.23508433097274167, 0.24461580357689838,
                     0.2769868768071587, 0.2831181875566533, 0.29026187858733815,
                     0.2933275881934319]
    let expIdxs = @[626, 46, 511, 110, 127, 248, 917, 530, 120, 656]
    let expRes = zip(expDists, expIdxs).sortedByIt(it[1])

    check nimRes.len == 10
    for i in 0 ..< 10:
      check nimRes[i][0] =~= expRes[i][0]
      check nimRes[i][1] == expRes[i][1]

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
    let nimRes = zip(nimResTup[0].toRawSeq,
                     nimResTup[1].toRawSeq).sortedByIt(it[1])

    let expDists = @[0.39204901822982824, 0.34227900996732774,
                     0.4669737528151749, 0.35602024612735655,
                     0.4249702365213115, 0.3064017509519704,
                     0.382091907998407, 0.45293765752691845,
                     0.3240002375605149, 0.43235987294945044,
                     0.40182889496693364, 0.33540945543963274,
                     0.3175277079140559, 0.3839873703517326,
                     0.3345738616476187, 0.3315389859698332,
                     0.23508433097274167, 0.24461580357689838,
                     0.3625861662346624, 0.19483515651264544,
                     0.2831181875566533, 0.4901321083399073,
                     0.3088205485838209, 0.39533061074509446,
                     0.4581916786054697, 0.4774790341208699,
                     0.33016321642086305, 0.4441578316520467,
                     0.43580912718555165, 0.49809495617847444,
                     0.4364540021674509, 0.4538118961873703,
                     0.29026187858733815, 0.2260793690650405,
                     0.34502009118405136, 0.42860258358652603,
                     0.14667821419698868, 0.22933987791539506,
                     0.403944873912106, 0.3703038832521505,
                     0.34774412778209185, 0.2769868768071587,
                     0.4953238287275528, 0.37094108327430014,
                     0.2933275881934319, 0.40434698209919323,
                     0.4069821720519212, 0.4847726254359657,
                     0.3808234425839364, 0.4398159214849333,
                     0.4992185062758554, 0.4908656707902833,
                     0.43737048571502907, 0.45278723835608875,
                     0.43147046746288176, 0.4560998063722978]
    let expIdxs = @[328, 667, 800, 631, 604, 72, 878, 276, 11, 570,
                    650, 582, 708, 659, 633, 395, 127, 248, 26, 46, 530,
                    909, 263, 498, 212, 554, 927, 271, 899, 567, 445,
                    635, 120, 511, 343, 797, 626, 110, 411, 988, 360,
                    917, 154, 292, 656, 123, 963, 938, 66, 942, 672, 264,
                    428, 549, 987, 402]
    let expRes = zip(expDists, expIdxs).sortedByIt(it[1])
    check nimRes.len == expRes.len
    for i in 0 ..< nimRes.len:
      check nimRes[i][0] =~= expRes[i][0]
      check nimRes[i][1] == expRes[i][1]

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
