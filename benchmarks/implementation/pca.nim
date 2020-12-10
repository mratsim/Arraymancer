import
  times, strformat,
  ../../src/arraymancer,
  ../../src/linear_algebra/helpers/auxiliary_blas

# Benchmarks of PCA implementations

# Helpers
# ---------------------------------------------------------------------------------
proc diag[T](d: Tensor[T], m, n: int): Tensor[T] {.noInit.}=
  # Creates a rectangular diagonal matrix
  assert d.rank == 1
  result = zeros[T](m, n)

  let k = min(m,n)
  assert d.size == k

  for i in 0 ..< k:
    result[i,i] = d[i]

proc mul_diag[T](x, diag: Tensor[T], K, N: int): Tensor[T] {.noInit.}=
  # Multiply x by diag, a rectangular diagonal matrix of shape [k, n]
  # represented by its diagonal only
  #
  #                |   a    0    0   0   0   |
  #                |   0    b    0   0   0   |
  #                |   0    0    c   0   0   |
  #
  # | x00 x01 x02|   ax00 bx01 cx02  0   0
  # | x10 x11 x12|   ax10 bx11 cx12  0   0
  # | x20 x21 x22|   ax20 bx21 cx22  0   0

  # Result in C order (row major)
  let M = x.shape[0]
  assert x.shape[1] == K

  result = zeros[T](M, N)
  let R = result.unsafe_raw_offset()
  let X = x.unsafe_raw_offset()
  let D = diag.unsafe_raw_offset()

  const Tile = 32
  let nk = min(N, K)

  # TODO parallel
  if x.is_F_contiguous:
    for i in countup(0, M-1, Tile):
      for j in countup(0, nk-1, Tile):
        for ii in i ..< min(i+Tile, M):
          for jj in j ..< min(j+Tile, nk):
            R[ii*N+jj] = X[jj*M+ii] * D[jj]
  elif x.is_C_contiguous:
    for i in countup(0, M-1, Tile):
      for j in countup(0, nk-1, Tile):
        for ii in i ..< min(i+Tile, M):
          for jj in j ..< min(j+Tile, nk):
            R[ii*N+jj] = X[ii*K+jj] * D[jj]

let a = [[1, 2, 3],
         [4, 5, 6],
         [7, 8, 9],
         [10, 11, 12]].toTensor()

let d = [[1, 0, 0, 0],
          [0, 2, 0, 0],
          [0, 0, 3, 0]].toTensor()

when false:
  doAssert d == [1,2,3].toTensor().diag(3, 4)
  doAssert a * d == a.mul_diag([1,2,3].toTensor().asContiguous(colMajor, force = true), 3, 4)
  doAssert a * d == a.mul_diag([1,2,3].toTensor(), 3, 4)
  # doAssert a * d == a *. [1,2,3].toTensor().unsqueeze(0)

  echo a * d
  # Tensor[system.int] of shape [4, 4]" on backend "Cpu"
  # |1      4       9       0|
  # |4      10      18      0|
  # |7      16      27      0|
  # |10     22      36      0|
  echo a *. [1,2,3].toTensor().unsqueeze(0)
  # Tensor[system.int] of shape [4, 3]" on backend "Cpu"
  # |1      4       9|
  # |4      10      18|
  # |7      16      27|
  # |10     22      36|

# # Implementations
# # ---------------------------------------------------------------------------------

proc pca_cov[T: SomeFloat](X: Tensor[T], n_components = 2): Tensor[T] {.noInit.}=
  # mean_centered
  let X = X -. X.mean(axis=0)
  let m = X.shape[0]
  let n = X.shape[1]

  # Compute covariance matrix
  var cov_matrix = newTensorUninit[T]([n, n])
  syrk(1.T / T(m-1), X, AtA, 0, cov_matrix, 'U')

  let (_, eigvecs) = cov_matrix.symeig(true, 'U', ^n_components .. ^1)

  let rotation_matrix = eigvecs[_, ^1..0|-1]
  result = X * rotation_matrix

proc pca_svd_tXU[T: SomeFloat](X: Tensor[T], n_components = 2): Tensor[T] {.noInit.}=
  let X = X -. X.mean(axis=0)

  let (U, _, _) = svd(X.transpose)
  result = X * U[_, 0..<n_components]

proc pca_svd_XV[T: SomeFloat](X: Tensor[T], n_components = 2): Tensor[T] {.noInit.}=
  let X = X -. X.mean(axis=0)

  let (_, _, Vh) = svd(X)
  result = X * Vh.transpose[_, 0..<n_components]

proc pca_svd_US[T: SomeFloat](X: Tensor[T], n_components = 2): Tensor[T] {.noInit.}=
  let X = X -. X.mean(axis=0)

  let (U, S, _) = svd(X)
  result = U[_, 0..<n_components] *. S[0..<n_components].unsqueeze(0)

proc pca_svd_tX_VhS[T: SomeFloat](X: Tensor[T], n_components = 2): Tensor[T] {.noInit.}=
  let X = X -. X.mean(axis=0)

  let (_, S, Vh) = svd(X.transpose)
  result = Vh.transpose[_, 0..<n_components] *. S[0..<n_components].unsqueeze(0)

# Sanity checks
# ---------------------------------------------------

block:
  let data = [[2.5, 2.4],
              [0.5, 0.7],
              [2.2, 2.9],
              [1.9, 2.2],
              [3.1, 3.0],
              [2.3, 2.7],
              [2.0, 1.6],
              [1.0, 1.1],
              [1.5, 1.6],
              [1.1, 0.9]].toTensor

  let expected = [[-0.827970186, -0.175115307],
                  [ 1.77758033,   0.142857227],
                  [-0.992197494,  0.384374989],
                  [-0.274210416,  0.130417207],
                  [-1.67580142,  -0.209498461],
                  [-0.912949103,  0.175282444],
                  [ 0.0991094375,-0.349824698],
                  [ 1.14457216,   0.0464172582],
                  [ 0.438046137,  0.0177646297],
                  [ 1.22382056,  -0.162675287]].toTensor

  let pca_cov = pca_cov(data, 2)
  let pca_svd_tXU = pca_svd_tXU(data, 2)
  let pca_svd_XV = pca_svd_XV(data, 2)
  let pca_svd_US = pca_svd_US(data, 2)
  let pca_svd_tX_VhS = pca_svd_tX_VhS(data, 2)

  for col in 0..<2:
    doAssert mean_absolute_error( pca_cov[_, col], expected[_, col]) < 1e-08 or
             mean_absolute_error(-pca_cov[_, col], expected[_, col]) < 1e-08

    doAssert mean_absolute_error( pca_svd_tXU[_, col], expected[_, col]) < 1e-08 or
             mean_absolute_error(-pca_svd_tXU[_, col], expected[_, col]) < 1e-08

    doAssert mean_absolute_error( pca_svd_XV[_, col], expected[_, col]) < 1e-08 or
             mean_absolute_error(-pca_svd_XV[_, col], expected[_, col]) < 1e-08

    doAssert mean_absolute_error( pca_svd_US[_, col], expected[_, col]) < 1e-08 or
             mean_absolute_error(-pca_svd_US[_, col], expected[_, col]) < 1e-08

    doAssert mean_absolute_error( pca_svd_tX_VhS[_, col], expected[_, col]) < 1e-08 or
             mean_absolute_error(-pca_svd_tX_VhS[_, col], expected[_, col]) < 1e-08

# Setup
# ---------------------------------------------------------------------------------

const k = 10
const Sizes = [100, 200, 400, 800, 1600, 3200]
const Display = false

type FloatType = float64

for Observations in Sizes:
  for Features in Sizes:
    let N = max(Observations, Features)

    echo "\n###########################"
    echo "Starting a new experiment\n"
    echo &"Matrix of shape: [{Observations}, {Features}]"
    echo &"Target PCA:      [{Observations}, {k}]\n"

    var start, stop: float64

    template profile(body: untyped): untyped {.dirty.} =
      start = epochTime()
      body
      stop = epochTime()

    template checkError(x, y: untyped) =
      let err = mean_absolute_error(x, y)
      let err_negy = mean_absolute_error(x, -y)
      let xStr = x.astToStr
      let yStr = y.astToStr
      # Strformat not working in templates grrr....

      # Note that signs are completely unstable so errors are not reliable
      # without a deterministic way to fix the signs
      echo "mean_absolute_error(", xStr, ", ", yStr,")  = ", err
      echo "mean_absolute_error(", xStr, ", -", yStr,") = ", err_negy
      echo "---------------------------------------------"

    # Create a known ill-conditionned matrix for testing
    let m = Observations
    let n = Features
    profile:
      let H = hilbert(N, FloatType)[0..<Observations, 0..<Features]
    echo &"Hilbert matrix creation took: {stop-start:>4.4f} seconds"

    profile:
      let pca_cov = pca_cov(H, k)
    echo &"pca_cov        took: {stop-start:>4.4f} seconds | [{n},{m}]*[{m},{n}] + symeig([{n},{n}])"

    profile:
      let pca_svd_tXU = pca_svd_tXU(H, k)
    echo &"pca_svd_tXU    took: {stop-start:>4.4f} seconds | svd([{n},{m}]) + [{m},{n}]*[{n},{k}]"

    profile:
      let pca_svd_XV = pca_svd_XV(H, k)
    echo &"pca_svd_XV     took: {stop-start:>4.4f} seconds | svd([{m},{n}]) + [{m},{n}]*[{n},{k}]"

    profile:
      let pca_svd_US = pca_svd_US(H, k)
    echo &"pca_svd_US     took: {stop-start:>4.4f} seconds | svd([{m},{n}]) + [{m},{k}]*[{k}]"

    profile:
      let pca_svd_tX_VhS = pca_svd_tX_VhS(H, k)
    echo &"pca_svd_tX_VhS took: {stop-start:>4.4f} seconds | svd([{n},{m}]) + [{n},{k}]*[{k}]"

    echo &"\nChecking that we have the same results."
    # We use pca_svd_US as a base as hilbert matrix is ill conditionned
    checkError(pca_svd_US, pca_cov)
    checkError(pca_svd_US, pca_svd_tXU)
    checkError(pca_svd_US, pca_svd_XV)
    checkError(pca_svd_US, pca_svd_tX_VhS)

    when Display:
      echo "\n--- pca_cov -------------------"
      echo pca_cov
      echo "\n--- pca_svd_tXU -------------------"
      echo pca_svd_tXU
      echo "\n--- pca_svd_XV -------------------"
      echo pca_svd_XV
      echo "\n--- pca_svd_US -------------------"
      echo pca_svd_tX_Vhs

# ###########################
# Starting a new experiment

# Matrix of shape: [100, 100]
# Target PCA:      [100, 10]

# Hilbert matrix creation took: 0.0001 seconds
# pca_cov        took: 0.0021 seconds | [100,100]*[100,100] + symeig([100,100])
# pca_svd_tXU    took: 0.0023 seconds | svd([100,100]) + [100,100]*[100,10]
# pca_svd_XV     took: 0.0022 seconds | svd([100,100]) + [100,100]*[100,10]
# pca_svd_US     took: 0.0022 seconds | svd([100,100]) + [100,10]*[10]
# pca_svd_tX_VhS took: 0.0021 seconds | svd([100,100]) + [100,10]*[10]

# Checking that we have the same results.
# mean_absolute_error(pca_svd_US, pca_cov)  = 0.001932655651739896
# mean_absolute_error(pca_svd_US, -pca_cov) = 0.02475863673565516
# ---------------------------------------------
# mean_absolute_error(pca_svd_US, pca_svd_tXU)  = 1.917320716572795e-17
# mean_absolute_error(pca_svd_US, -pca_svd_tXU) = 0.02669129238729377
# ---------------------------------------------
# mean_absolute_error(pca_svd_US, pca_svd_XV)  = 3.72577998266549e-17
# mean_absolute_error(pca_svd_US, -pca_svd_XV) = 0.02669129238729376
# ---------------------------------------------
# mean_absolute_error(pca_svd_US, pca_svd_tX_VhS)  = 2.911447131399173e-17
# mean_absolute_error(pca_svd_US, -pca_svd_tX_VhS) = 0.02669129238729375
# ---------------------------------------------

# ###########################
# Starting a new experiment

# Matrix of shape: [100, 200]
# Target PCA:      [100, 10]

# Hilbert matrix creation took: 0.0001 seconds
# pca_cov        took: 0.0050 seconds | [200,100]*[100,200] + symeig([200,200])
# pca_svd_tXU    took: 0.0301 seconds | svd([200,100]) + [100,200]*[200,10]
# pca_svd_XV     took: 0.0060 seconds | svd([100,200]) + [100,200]*[200,10]
# pca_svd_US     took: 0.0056 seconds | svd([100,200]) + [100,10]*[10]
# pca_svd_tX_VhS took: 0.0050 seconds | svd([200,100]) + [200,10]*[10]

# Checking that we have the same results.
# mean_absolute_error(pca_svd_US, pca_cov)  = 0.002030719975565127
# mean_absolute_error(pca_svd_US, -pca_cov) = 0.0248767720101967
# ---------------------------------------------
# mean_absolute_error(pca_svd_US, pca_svd_tXU)  = 0.008595519202340833
# mean_absolute_error(pca_svd_US, -pca_svd_tXU) = 0.01831197278331903
# ---------------------------------------------
# mean_absolute_error(pca_svd_US, pca_svd_XV)  = 2.845859912353192e-17
# mean_absolute_error(pca_svd_US, -pca_svd_XV) = 0.02690749198565989
# ---------------------------------------------
# mean_absolute_error(pca_svd_US, pca_svd_tX_VhS)  = 0.008595519202340834
# mean_absolute_error(pca_svd_US, -pca_svd_tX_VhS) = 0.01831197278331902
# ---------------------------------------------

# ###########################
# Starting a new experiment

# Matrix of shape: [100, 400]
# Target PCA:      [100, 10]

# Hilbert matrix creation took: 0.0007 seconds
# pca_cov        took: 0.0318 seconds | [400,100]*[100,400] + symeig([400,400])
# pca_svd_tXU    took: 0.0058 seconds | svd([400,100]) + [100,400]*[400,10]
# pca_svd_XV     took: 0.0092 seconds | svd([100,400]) + [100,400]*[400,10]
# pca_svd_US     took: 0.0084 seconds | svd([100,400]) + [100,10]*[10]
# pca_svd_tX_VhS took: 0.0053 seconds | svd([400,100]) + [400,10]*[10]

# Checking that we have the same results.
# mean_absolute_error(pca_svd_US, pca_cov)  = 0.002059295033802309
# mean_absolute_error(pca_svd_US, -pca_cov) = 0.02490518148768566
# ---------------------------------------------
# mean_absolute_error(pca_svd_US, pca_svd_tXU)  = 0.008645474850949521
# mean_absolute_error(pca_svd_US, -pca_svd_tXU) = 0.01831900167046272
# ---------------------------------------------
# mean_absolute_error(pca_svd_US, pca_svd_XV)  = 3.297168629581948e-17
# mean_absolute_error(pca_svd_US, -pca_svd_XV) = 0.02696447652141224
# ---------------------------------------------
# mean_absolute_error(pca_svd_US, pca_svd_tX_VhS)  = 0.008645474850949516
# mean_absolute_error(pca_svd_US, -pca_svd_tX_VhS) = 0.01831900167046272
# ---------------------------------------------

# ###########################
# Starting a new experiment

# Matrix of shape: [100, 800]
# Target PCA:      [100, 10]

# Hilbert matrix creation took: 0.0023 seconds
# pca_cov        took: 0.0712 seconds | [800,100]*[100,800] + symeig([800,800])
# pca_svd_tXU    took: 0.0060 seconds | svd([800,100]) + [100,800]*[800,10]
# pca_svd_XV     took: 0.0239 seconds | svd([100,800]) + [100,800]*[800,10]
# pca_svd_US     took: 0.0207 seconds | svd([100,800]) + [100,10]*[10]
# pca_svd_tX_VhS took: 0.0058 seconds | svd([800,100]) + [800,10]*[10]

# Checking that we have the same results.
# mean_absolute_error(pca_svd_US, pca_cov)  = 0.002065366626663964
# mean_absolute_error(pca_svd_US, -pca_cov) = 0.02491079550723915
# ---------------------------------------------
# mean_absolute_error(pca_svd_US, pca_svd_tXU)  = 0.008656026307183727
# mean_absolute_error(pca_svd_US, -pca_svd_tXU) = 0.01832013582664856
# ---------------------------------------------
# mean_absolute_error(pca_svd_US, pca_svd_XV)  = 3.882898581731306e-17
# mean_absolute_error(pca_svd_US, -pca_svd_XV) = 0.02697616213383223
# ---------------------------------------------
# mean_absolute_error(pca_svd_US, pca_svd_tX_VhS)  = 0.008656026307183718
# mean_absolute_error(pca_svd_US, -pca_svd_tX_VhS) = 0.01832013582664856
# ---------------------------------------------

# ###########################
# Starting a new experiment

# Matrix of shape: [100, 1600]
# Target PCA:      [100, 10]

# Hilbert matrix creation took: 0.0099 seconds
# pca_cov        took: 0.2151 seconds | [1600,100]*[100,1600] + symeig([1600,1600])
# pca_svd_tXU    took: 0.0078 seconds | svd([1600,100]) + [100,1600]*[1600,10]
# pca_svd_XV     took: 0.0330 seconds | svd([100,1600]) + [100,1600]*[1600,10]
# pca_svd_US     took: 0.0246 seconds | svd([100,1600]) + [100,10]*[10]
# pca_svd_tX_VhS took: 0.0134 seconds | svd([1600,100]) + [1600,10]*[10]

# Checking that we have the same results.
# mean_absolute_error(pca_svd_US, pca_cov)  = 0.002066398351083621
# mean_absolute_error(pca_svd_US, -pca_cov) = 0.02491173412596473
# ---------------------------------------------
# mean_absolute_error(pca_svd_US, pca_svd_tXU)  = 0.008657793701796309
# mean_absolute_error(pca_svd_US, -pca_svd_tXU) = 0.01832033877518993
# ---------------------------------------------
# mean_absolute_error(pca_svd_US, pca_svd_XV)  = 4.353824274989707e-17
# mean_absolute_error(pca_svd_US, -pca_svd_XV) = 0.02697813247698625
# ---------------------------------------------
# mean_absolute_error(pca_svd_US, pca_svd_tX_VhS)  = 0.008657793701796302
# mean_absolute_error(pca_svd_US, -pca_svd_tX_VhS) = 0.01832033877518993
# ---------------------------------------------

# ###########################
# Starting a new experiment

# Matrix of shape: [100, 3200]
# Target PCA:      [100, 10]

# Hilbert matrix creation took: 0.0413 seconds
# pca_cov        took: 0.7771 seconds | [3200,100]*[100,3200] + symeig([3200,3200])
# pca_svd_tXU    took: 0.0197 seconds | svd([3200,100]) + [100,3200]*[3200,10]
# pca_svd_XV     took: 0.0377 seconds | svd([100,3200]) + [100,3200]*[3200,10]
# pca_svd_US     took: 0.0409 seconds | svd([100,3200]) + [100,10]*[10]
# pca_svd_tX_VhS took: 0.0224 seconds | svd([3200,100]) + [3200,10]*[10]

# Checking that we have the same results.
# mean_absolute_error(pca_svd_US, pca_cov)  = 0.002066551364076294
# mean_absolute_error(pca_svd_US, -pca_cov) = 0.02491187312200345
# ---------------------------------------------
# mean_absolute_error(pca_svd_US, pca_svd_tXU)  = 0.008658055773879248
# mean_absolute_error(pca_svd_US, -pca_svd_tXU) = 0.01832036871212876
# ---------------------------------------------
# mean_absolute_error(pca_svd_US, pca_svd_XV)  = 4.767932554486612e-17
# mean_absolute_error(pca_svd_US, -pca_svd_XV) = 0.02697842448600803
# ---------------------------------------------
# mean_absolute_error(pca_svd_US, pca_svd_tX_VhS)  = 0.008658055773879257
# mean_absolute_error(pca_svd_US, -pca_svd_tX_VhS) = 0.01832036871212876
# ---------------------------------------------

# ###########################
# Starting a new experiment

# Matrix of shape: [200, 100]
# Target PCA:      [200, 10]

# Hilbert matrix creation took: 0.0000 seconds
# pca_cov        took: 0.0017 seconds | [100,200]*[200,100] + symeig([100,100])
# pca_svd_tXU    took: 0.0061 seconds | svd([100,200]) + [200,100]*[100,10]
# pca_svd_XV     took: 0.0046 seconds | svd([200,100]) + [200,100]*[100,10]
# pca_svd_US     took: 0.0042 seconds | svd([200,100]) + [200,10]*[10]
# pca_svd_tX_VhS took: 0.0056 seconds | svd([100,200]) + [100,10]*[10]

# Checking that we have the same results.
# mean_absolute_error(pca_svd_US, pca_cov)  = 0.005370049268789392
# mean_absolute_error(pca_svd_US, -pca_cov) = 0.01468008482339707
# ---------------------------------------------
# mean_absolute_error(pca_svd_US, pca_svd_tXU)  = 0.007155425293366204
# mean_absolute_error(pca_svd_US, -pca_svd_tXU) = 0.01289470879879414
# ---------------------------------------------
# mean_absolute_error(pca_svd_US, pca_svd_XV)  = 1.798122214986902e-17
# mean_absolute_error(pca_svd_US, -pca_svd_XV) = 0.0200501340921603
# ---------------------------------------------
# mean_absolute_error(pca_svd_US, pca_svd_tX_VhS)  = 0.007155425293366199
# mean_absolute_error(pca_svd_US, -pca_svd_tX_VhS) = 0.01289470879879414
# ---------------------------------------------

# ###########################
# Starting a new experiment

# Matrix of shape: [200, 200]
# Target PCA:      [200, 10]

# Hilbert matrix creation took: 0.0000 seconds
# pca_cov        took: 0.0052 seconds | [200,200]*[200,200] + symeig([200,200])
# pca_svd_tXU    took: 0.0084 seconds | svd([200,200]) + [200,200]*[200,10]
# pca_svd_XV     took: 0.0302 seconds | svd([200,200]) + [200,200]*[200,10]
# pca_svd_US     took: 0.0080 seconds | svd([200,200]) + [200,10]*[10]
# pca_svd_tX_VhS took: 0.0080 seconds | svd([200,200]) + [200,10]*[10]

# Checking that we have the same results.
# mean_absolute_error(pca_svd_US, pca_cov)  = 0.002012538895062467
# mean_absolute_error(pca_svd_US, -pca_cov) = 0.01847800800226472
# ---------------------------------------------
# mean_absolute_error(pca_svd_US, pca_svd_tXU)  = 2.106376006293534e-17
# mean_absolute_error(pca_svd_US, -pca_svd_tXU) = 0.02049054689731615
# ---------------------------------------------
# mean_absolute_error(pca_svd_US, pca_svd_XV)  = 1.841540617916843e-17
# mean_absolute_error(pca_svd_US, -pca_svd_XV) = 0.02049054689731615
# ---------------------------------------------
# mean_absolute_error(pca_svd_US, pca_svd_tX_VhS)  = 1.965713397019082e-17
# mean_absolute_error(pca_svd_US, -pca_svd_tX_VhS) = 0.02049054689731615
# ---------------------------------------------

# ###########################
# Starting a new experiment

# Matrix of shape: [200, 400]
# Target PCA:      [200, 10]

# Hilbert matrix creation took: 0.0002 seconds
# pca_cov        took: 0.0350 seconds | [400,200]*[200,400] + symeig([400,400])
# pca_svd_tXU    took: 0.0137 seconds | svd([400,200]) + [200,400]*[400,10]
# pca_svd_XV     took: 0.0200 seconds | svd([200,400]) + [200,400]*[400,10]
# pca_svd_US     took: 0.0206 seconds | svd([200,400]) + [200,10]*[10]
# pca_svd_tX_VhS took: 0.0150 seconds | svd([400,200]) + [400,10]*[10]

# Checking that we have the same results.
# mean_absolute_error(pca_svd_US, pca_cov)  = 0.001655813037479778
# mean_absolute_error(pca_svd_US, -pca_cov) = 0.01898928510276307
# ---------------------------------------------
# mean_absolute_error(pca_svd_US, pca_svd_tXU)  = 0.00719962326591146
# mean_absolute_error(pca_svd_US, -pca_svd_tXU) = 0.01344547487432698
# ---------------------------------------------
# mean_absolute_error(pca_svd_US, pca_svd_XV)  = 3.285345463476061e-17
# mean_absolute_error(pca_svd_US, -pca_svd_XV) = 0.02064509814023848
# ---------------------------------------------
# mean_absolute_error(pca_svd_US, pca_svd_tX_VhS)  = 0.007199623265911457
# mean_absolute_error(pca_svd_US, -pca_svd_tX_VhS) = 0.01344547487432698
# ---------------------------------------------

# ###########################
# Starting a new experiment

# Matrix of shape: [200, 800]
# Target PCA:      [200, 10]

# Hilbert matrix creation took: 0.0010 seconds
# pca_cov        took: 0.0714 seconds | [800,200]*[200,800] + symeig([800,800])
# pca_svd_tXU    took: 0.0177 seconds | svd([800,200]) + [200,800]*[800,10]
# pca_svd_XV     took: 0.0357 seconds | svd([200,800]) + [200,800]*[800,10]
# pca_svd_US     took: 0.0331 seconds | svd([200,800]) + [200,10]*[10]
# pca_svd_tX_VhS took: 0.0221 seconds | svd([800,200]) + [800,10]*[10]

# Checking that we have the same results.
# mean_absolute_error(pca_svd_US, pca_cov)  = 0.001669777155822621
# mean_absolute_error(pca_svd_US, -pca_cov) = 0.01901603180905142
# ---------------------------------------------
# mean_absolute_error(pca_svd_US, pca_svd_tXU)  = 0.007231722364766706
# mean_absolute_error(pca_svd_US, -pca_svd_tXU) = 0.0134540866000991
# ---------------------------------------------
# mean_absolute_error(pca_svd_US, pca_svd_XV)  = 4.452961897481056e-17
# mean_absolute_error(pca_svd_US, -pca_svd_XV) = 0.02068580896486581
# ---------------------------------------------
# mean_absolute_error(pca_svd_US, pca_svd_tX_VhS)  = 0.007231722364766708
# mean_absolute_error(pca_svd_US, -pca_svd_tX_VhS) = 0.0134540866000991
# ---------------------------------------------

# ###########################
# Starting a new experiment

# Matrix of shape: [200, 1600]
# Target PCA:      [200, 10]

# Hilbert matrix creation took: 0.0044 seconds
# pca_cov        took: 0.1992 seconds | [1600,200]*[200,1600] + symeig([1600,1600])
# pca_svd_tXU    took: 0.0379 seconds | svd([1600,200]) + [200,1600]*[1600,10]
# pca_svd_XV     took: 0.0498 seconds | svd([200,1600]) + [200,1600]*[1600,10]
# pca_svd_US     took: 0.0444 seconds | svd([200,1600]) + [200,10]*[10]
# pca_svd_tX_VhS took: 0.0262 seconds | svd([1600,200]) + [1600,10]*[10]

# Checking that we have the same results.
# mean_absolute_error(pca_svd_US, pca_cov)  = 0.001672761361514709
# mean_absolute_error(pca_svd_US, -pca_cov) = 0.01902142018685975
# ---------------------------------------------
# mean_absolute_error(pca_svd_US, pca_svd_tXU)  = 0.007236956723672422
# mean_absolute_error(pca_svd_US, -pca_svd_tXU) = 0.01345722482469422
# ---------------------------------------------
# mean_absolute_error(pca_svd_US, pca_svd_XV)  = 4.751903145156371e-17
# mean_absolute_error(pca_svd_US, -pca_svd_XV) = 0.0206941815483666
# ---------------------------------------------
# mean_absolute_error(pca_svd_US, pca_svd_tX_VhS)  = 0.007236956723672422
# mean_absolute_error(pca_svd_US, -pca_svd_tX_VhS) = 0.01345722482469421
# ---------------------------------------------

# ###########################
# Starting a new experiment

# Matrix of shape: [200, 3200]
# Target PCA:      [200, 10]

# Hilbert matrix creation took: 0.1137 seconds
# pca_cov        took: 0.8398 seconds | [3200,200]*[200,3200] + symeig([3200,3200])
# pca_svd_tXU    took: 0.0356 seconds | svd([3200,200]) + [200,3200]*[3200,10]
# pca_svd_XV     took: 0.0683 seconds | svd([200,3200]) + [200,3200]*[3200,10]
# pca_svd_US     took: 0.1462 seconds | svd([200,3200]) + [200,10]*[10]
# pca_svd_tX_VhS took: 0.0359 seconds | svd([3200,200]) + [3200,10]*[10]

# Checking that we have the same results.
# mean_absolute_error(pca_svd_US, pca_cov)  = 0.001673281525307154
# mean_absolute_error(pca_svd_US, -pca_cov) = 0.01902231420210297
# ---------------------------------------------
# mean_absolute_error(pca_svd_US, pca_svd_tXU)  = 0.007237829221379827
# mean_absolute_error(pca_svd_US, -pca_svd_tXU) = 0.01345776650602605
# ---------------------------------------------
# mean_absolute_error(pca_svd_US, pca_svd_XV)  = 5.270110067340357e-17
# mean_absolute_error(pca_svd_US, -pca_svd_XV) = 0.02069559572740585
# ---------------------------------------------
# mean_absolute_error(pca_svd_US, pca_svd_tX_VhS)  = 0.007237829221379825
# mean_absolute_error(pca_svd_US, -pca_svd_tX_VhS) = 0.01345776650602604
# ---------------------------------------------

# ###########################
# Starting a new experiment

# Matrix of shape: [400, 100]
# Target PCA:      [400, 10]

# Hilbert matrix creation took: 0.0004 seconds
# pca_cov        took: 0.0019 seconds | [100,400]*[400,100] + symeig([100,100])
# pca_svd_tXU    took: 0.0086 seconds | svd([100,400]) + [400,100]*[100,10]
# pca_svd_XV     took: 0.0055 seconds | svd([400,100]) + [400,100]*[100,10]
# pca_svd_US     took: 0.0055 seconds | svd([400,100]) + [400,10]*[10]
# pca_svd_tX_VhS took: 0.0085 seconds | svd([100,400]) + [100,10]*[10]

# Checking that we have the same results.
# mean_absolute_error(pca_svd_US, pca_cov)  = 0.004140240515741197
# mean_absolute_error(pca_svd_US, -pca_cov) = 0.01035599272945293
# ---------------------------------------------
# mean_absolute_error(pca_svd_US, pca_svd_tXU)  = 0.005323374186186267
# mean_absolute_error(pca_svd_US, -pca_svd_tXU) = 0.009172859058940585
# ---------------------------------------------
# mean_absolute_error(pca_svd_US, pca_svd_XV)  = 7.05585684725914e-18
# mean_absolute_error(pca_svd_US, -pca_svd_XV) = 0.01449623324512687
# ---------------------------------------------
# mean_absolute_error(pca_svd_US, pca_svd_tX_VhS)  = 0.005323374186186267
# mean_absolute_error(pca_svd_US, -pca_svd_tX_VhS) = 0.009172859058940584
# ---------------------------------------------

# ###########################
# Starting a new experiment

# Matrix of shape: [400, 200]
# Target PCA:      [400, 10]

# Hilbert matrix creation took: 0.0002 seconds
# pca_cov        took: 0.0060 seconds | [200,400]*[400,200] + symeig([200,200])
# pca_svd_tXU    took: 0.0380 seconds | svd([200,400]) + [400,200]*[200,10]
# pca_svd_XV     took: 0.0164 seconds | svd([400,200]) + [400,200]*[200,10]
# pca_svd_US     took: 0.0282 seconds | svd([400,200]) + [400,10]*[10]
# pca_svd_tX_VhS took: 0.0269 seconds | svd([200,400]) + [200,10]*[10]

# Checking that we have the same results.
# mean_absolute_error(pca_svd_US, pca_cov)  = 0.00441437668940445
# mean_absolute_error(pca_svd_US, -pca_cov) = 0.01076649970932751
# ---------------------------------------------
# mean_absolute_error(pca_svd_US, pca_svd_tXU)  = 0.005769091577369347
# mean_absolute_error(pca_svd_US, -pca_svd_tXU) = 0.009411784821342098
# ---------------------------------------------
# mean_absolute_error(pca_svd_US, pca_svd_XV)  = 1.035977586746714e-17
# mean_absolute_error(pca_svd_US, -pca_svd_XV) = 0.01518087639871145
# ---------------------------------------------
# mean_absolute_error(pca_svd_US, pca_svd_tX_VhS)  = 0.005769091577369343
# mean_absolute_error(pca_svd_US, -pca_svd_tX_VhS) = 0.009411784821342093
# ---------------------------------------------

# ###########################
# Starting a new experiment

# Matrix of shape: [400, 400]
# Target PCA:      [400, 10]

# Hilbert matrix creation took: 0.0002 seconds
# pca_cov        took: 0.0289 seconds | [400,400]*[400,400] + symeig([400,400])
# pca_svd_tXU    took: 0.0479 seconds | svd([400,400]) + [400,400]*[400,10]
# pca_svd_XV     took: 0.0454 seconds | svd([400,400]) + [400,400]*[400,10]
# pca_svd_US     took: 0.0335 seconds | svd([400,400]) + [400,10]*[10]
# pca_svd_tX_VhS took: 0.0351 seconds | svd([400,400]) + [400,10]*[10]

# Checking that we have the same results.
# mean_absolute_error(pca_svd_US, pca_cov)  = 0.001500797345135291
# mean_absolute_error(pca_svd_US, -pca_cov) = 0.01399845249749339
# ---------------------------------------------
# mean_absolute_error(pca_svd_US, pca_svd_tXU)  = 2.050789799018646e-17
# mean_absolute_error(pca_svd_US, -pca_svd_tXU) = 0.01549924984262063
# ---------------------------------------------
# mean_absolute_error(pca_svd_US, pca_svd_XV)  = 1.962018239540985e-17
# mean_absolute_error(pca_svd_US, -pca_svd_XV) = 0.01549924984262063
# ---------------------------------------------
# mean_absolute_error(pca_svd_US, pca_svd_tX_VhS)  = 2.172027137967178e-17
# mean_absolute_error(pca_svd_US, -pca_svd_tX_VhS) = 0.01549924984262063
# ---------------------------------------------

# ###########################
# Starting a new experiment

# Matrix of shape: [400, 800]
# Target PCA:      [400, 10]

# Hilbert matrix creation took: 0.0010 seconds
# pca_cov        took: 0.0860 seconds | [800,400]*[400,800] + symeig([800,800])
# pca_svd_tXU    took: 0.0594 seconds | svd([800,400]) + [400,800]*[800,10]
# pca_svd_XV     took: 0.0724 seconds | svd([400,800]) + [400,800]*[800,10]
# pca_svd_US     took: 0.0765 seconds | svd([400,800]) + [400,10]*[10]
# pca_svd_tX_VhS took: 0.0525 seconds | svd([800,400]) + [800,10]*[10]

# Checking that we have the same results.
# mean_absolute_error(pca_svd_US, pca_cov)  = 0.001537648550348399
# mean_absolute_error(pca_svd_US, -pca_cov) = 0.01407275521655781
# ---------------------------------------------
# mean_absolute_error(pca_svd_US, pca_svd_tXU)  = 0.00608612507850304
# mean_absolute_error(pca_svd_US, -pca_svd_tXU) = 0.009524278688396207
# ---------------------------------------------
# mean_absolute_error(pca_svd_US, pca_svd_XV)  = 1.724823702007888e-17
# mean_absolute_error(pca_svd_US, -pca_svd_XV) = 0.0156104037668993
# ---------------------------------------------
# mean_absolute_error(pca_svd_US, pca_svd_tX_VhS)  = 0.006086125078503029
# mean_absolute_error(pca_svd_US, -pca_svd_tX_VhS) = 0.009524278688396205
# ---------------------------------------------

# ###########################
# Starting a new experiment

# Matrix of shape: [400, 1600]
# Target PCA:      [400, 10]

# Hilbert matrix creation took: 0.0043 seconds
# pca_cov        took: 0.2015 seconds | [1600,400]*[400,1600] + symeig([1600,1600])
# pca_svd_tXU    took: 0.0712 seconds | svd([1600,400]) + [400,1600]*[1600,10]
# pca_svd_XV     took: 0.0937 seconds | svd([400,1600]) + [400,1600]*[1600,10]
# pca_svd_US     took: 0.0957 seconds | svd([400,1600]) + [400,10]*[10]
# pca_svd_tX_VhS took: 0.0602 seconds | svd([1600,400]) + [1600,10]*[10]

# Checking that we have the same results.
# mean_absolute_error(pca_svd_US, pca_cov)  = 0.001547844608095407
# mean_absolute_error(pca_svd_US, -pca_cov) = 0.01409169028277872
# ---------------------------------------------
# mean_absolute_error(pca_svd_US, pca_svd_tXU)  = 0.006106473938051494
# mean_absolute_error(pca_svd_US, -pca_svd_tXU) = 0.009533060952816532
# ---------------------------------------------
# mean_absolute_error(pca_svd_US, pca_svd_XV)  = 2.543334743921779e-17
# mean_absolute_error(pca_svd_US, -pca_svd_XV) = 0.01563953489086801
# ---------------------------------------------
# mean_absolute_error(pca_svd_US, pca_svd_tX_VhS)  = 0.006106473938051486
# mean_absolute_error(pca_svd_US, -pca_svd_tX_VhS) = 0.009533060952816528
# ---------------------------------------------

# ###########################
# Starting a new experiment

# Matrix of shape: [400, 3200]
# Target PCA:      [400, 10]

# Hilbert matrix creation took: 0.0324 seconds
# pca_cov        took: 0.7153 seconds | [3200,400]*[400,3200] + symeig([3200,3200])
# pca_svd_tXU    took: 0.0822 seconds | svd([3200,400]) + [400,3200]*[3200,10]
# pca_svd_XV     took: 0.1429 seconds | svd([400,3200]) + [400,3200]*[3200,10]
# pca_svd_US     took: 0.1427 seconds | svd([400,3200]) + [400,10]*[10]
# pca_svd_tX_VhS took: 0.0805 seconds | svd([3200,400]) + [3200,10]*[10]

# Checking that we have the same results.
# mean_absolute_error(pca_svd_US, pca_cov)  = 0.001550019737760012
# mean_absolute_error(pca_svd_US, -pca_cov) = 0.01409548817693593
# ---------------------------------------------
# mean_absolute_error(pca_svd_US, pca_svd_tXU)  = 0.006110732239280143
# mean_absolute_error(pca_svd_US, -pca_svd_tXU) = 0.009534775675410402
# ---------------------------------------------
# mean_absolute_error(pca_svd_US, pca_svd_XV)  = 2.242829877892608e-17
# mean_absolute_error(pca_svd_US, -pca_svd_XV) = 0.0156455079146906
# ---------------------------------------------
# mean_absolute_error(pca_svd_US, pca_svd_tX_VhS)  = 0.006110732239280132
# mean_absolute_error(pca_svd_US, -pca_svd_tX_VhS) = 0.009534775675410397
# ---------------------------------------------

# ###########################
# Starting a new experiment

# Matrix of shape: [800, 100]
# Target PCA:      [800, 10]

# Hilbert matrix creation took: 0.0010 seconds
# pca_cov        took: 0.0025 seconds | [100,800]*[800,100] + symeig([100,100])
# pca_svd_tXU    took: 0.0333 seconds | svd([100,800]) + [800,100]*[100,10]
# pca_svd_XV     took: 0.0070 seconds | svd([800,100]) + [800,100]*[100,10]
# pca_svd_US     took: 0.0056 seconds | svd([800,100]) + [800,10]*[10]
# pca_svd_tX_VhS took: 0.0229 seconds | svd([100,800]) + [100,10]*[10]

# Checking that we have the same results.
# mean_absolute_error(pca_svd_US, pca_cov)  = 0.003010258301095772
# mean_absolute_error(pca_svd_US, -pca_cov) = 0.007059520450906202
# ---------------------------------------------
# mean_absolute_error(pca_svd_US, pca_svd_tXU)  = 0.003901973528120972
# mean_absolute_error(pca_svd_US, -pca_svd_tXU) = 0.006167805223862379
# ---------------------------------------------
# mean_absolute_error(pca_svd_US, pca_svd_XV)  = 7.286513129037402e-18
# mean_absolute_error(pca_svd_US, -pca_svd_XV) = 0.01006977875198341
# ---------------------------------------------
# mean_absolute_error(pca_svd_US, pca_svd_tX_VhS)  = 0.00390197352812097
# mean_absolute_error(pca_svd_US, -pca_svd_tX_VhS) = 0.006167805223862385
# ---------------------------------------------

# ###########################
# Starting a new experiment

# Matrix of shape: [800, 200]
# Target PCA:      [800, 10]

# Hilbert matrix creation took: 0.0011 seconds
# pca_cov        took: 0.0125 seconds | [200,800]*[800,200] + symeig([200,200])
# pca_svd_tXU    took: 0.0392 seconds | svd([200,800]) + [800,200]*[200,10]
# pca_svd_XV     took: 0.0143 seconds | svd([800,200]) + [800,200]*[200,10]
# pca_svd_US     took: 0.0158 seconds | svd([800,200]) + [800,10]*[10]
# pca_svd_tX_VhS took: 0.0256 seconds | svd([200,800]) + [200,10]*[10]

# Checking that we have the same results.
# mean_absolute_error(pca_svd_US, pca_cov)  = 0.003336269680585353
# mean_absolute_error(pca_svd_US, -pca_cov) = 0.007555221933793573
# ---------------------------------------------
# mean_absolute_error(pca_svd_US, pca_svd_tXU)  = 0.004414832142840589
# mean_absolute_error(pca_svd_US, -pca_svd_tXU) = 0.006476659471528319
# ---------------------------------------------
# mean_absolute_error(pca_svd_US, pca_svd_XV)  = 1.053456752967794e-17
# mean_absolute_error(pca_svd_US, -pca_svd_XV) = 0.01089149161436891
# ---------------------------------------------
# mean_absolute_error(pca_svd_US, pca_svd_tX_VhS)  = 0.004414832142840589
# mean_absolute_error(pca_svd_US, -pca_svd_tX_VhS) = 0.006476659471528319
# ---------------------------------------------

# ###########################
# Starting a new experiment

# Matrix of shape: [800, 400]
# Target PCA:      [800, 10]

# Hilbert matrix creation took: 0.0010 seconds
# pca_cov        took: 0.0424 seconds | [400,800]*[800,400] + symeig([400,400])
# pca_svd_tXU    took: 0.0846 seconds | svd([400,800]) + [800,400]*[400,10]
# pca_svd_XV     took: 0.0482 seconds | svd([800,400]) + [800,400]*[400,10]
# pca_svd_US     took: 0.0494 seconds | svd([800,400]) + [800,10]*[10]
# pca_svd_tX_VhS took: 0.0763 seconds | svd([400,800]) + [400,10]*[10]

# Checking that we have the same results.
# mean_absolute_error(pca_svd_US, pca_cov)  = 0.0034998034813876
# mean_absolute_error(pca_svd_US, -pca_cov) = 0.007883069342978328
# ---------------------------------------------
# mean_absolute_error(pca_svd_US, pca_svd_tXU)  = 0.00476235467422674
# mean_absolute_error(pca_svd_US, -pca_svd_tXU) = 0.006620518150134859
# ---------------------------------------------
# mean_absolute_error(pca_svd_US, pca_svd_XV)  = 1.360647785467498e-17
# mean_absolute_error(pca_svd_US, -pca_svd_XV) = 0.01138287282436162
# ---------------------------------------------
# mean_absolute_error(pca_svd_US, pca_svd_tX_VhS)  = 0.004762354674226725
# mean_absolute_error(pca_svd_US, -pca_svd_tX_VhS) = 0.006620518150134863
# ---------------------------------------------

# ###########################
# Starting a new experiment

# Matrix of shape: [800, 800]
# Target PCA:      [800, 10]

# Hilbert matrix creation took: 0.0010 seconds
# pca_cov        took: 0.0865 seconds | [800,800]*[800,800] + symeig([800,800])
# pca_svd_tXU    took: 0.1420 seconds | svd([800,800]) + [800,800]*[800,10]
# pca_svd_XV     took: 0.1279 seconds | svd([800,800]) + [800,800]*[800,10]
# pca_svd_US     took: 0.1342 seconds | svd([800,800]) + [800,10]*[10]
# pca_svd_tX_VhS took: 0.1289 seconds | svd([800,800]) + [800,10]*[10]

# Checking that we have the same results.
# mean_absolute_error(pca_svd_US, pca_cov)  = 0.001324183122544304
# mean_absolute_error(pca_svd_US, -pca_cov) = 0.01028733842504989
# ---------------------------------------------
# mean_absolute_error(pca_svd_US, pca_svd_tXU)  = 2.27028088922561e-17
# mean_absolute_error(pca_svd_US, -pca_svd_tXU) = 0.01161152154759139
# ---------------------------------------------
# mean_absolute_error(pca_svd_US, pca_svd_XV)  = 2.739215337757956e-17
# mean_absolute_error(pca_svd_US, -pca_svd_XV) = 0.01161152154759139
# ---------------------------------------------
# mean_absolute_error(pca_svd_US, pca_svd_tX_VhS)  = 2.544326209283469e-17
# mean_absolute_error(pca_svd_US, -pca_svd_tX_VhS) = 0.01161152154759138
# ---------------------------------------------

# ###########################
# Starting a new experiment

# Matrix of shape: [800, 1600]
# Target PCA:      [800, 10]

# Hilbert matrix creation took: 0.0044 seconds
# pca_cov        took: 0.1970 seconds | [1600,800]*[800,1600] + symeig([1600,1600])
# pca_svd_tXU    took: 0.2077 seconds | svd([1600,800]) + [800,1600]*[1600,10]
# pca_svd_XV     took: 0.2490 seconds | svd([800,1600]) + [800,1600]*[1600,10]
# pca_svd_US     took: 0.2775 seconds | svd([800,1600]) + [800,10]*[10]
# pca_svd_tX_VhS took: 0.1785 seconds | svd([1600,800]) + [1600,10]*[10]

# Checking that we have the same results.
# mean_absolute_error(pca_svd_US, pca_cov)  = 0.001349796772436461
# mean_absolute_error(pca_svd_US, -pca_cov) = 0.01034190739810298
# ---------------------------------------------
# mean_absolute_error(pca_svd_US, pca_svd_tXU)  = 0.004957763307770388
# mean_absolute_error(pca_svd_US, -pca_svd_tXU) = 0.00673394086276667
# ---------------------------------------------
# mean_absolute_error(pca_svd_US, pca_svd_XV)  = 1.719855696791456e-17
# mean_absolute_error(pca_svd_US, -pca_svd_XV) = 0.01169170417053698
# ---------------------------------------------
# mean_absolute_error(pca_svd_US, pca_svd_tX_VhS)  = 0.004957763307770376
# mean_absolute_error(pca_svd_US, -pca_svd_tX_VhS) = 0.006733940862766659
# ---------------------------------------------

# ###########################
# Starting a new experiment

# Matrix of shape: [800, 3200]
# Target PCA:      [800, 10]

# Hilbert matrix creation took: 0.0706 seconds
# pca_cov        took: 0.8422 seconds | [3200,800]*[800,3200] + symeig([3200,3200])
# pca_svd_tXU    took: 0.2789 seconds | svd([3200,800]) + [800,3200]*[3200,10]
# pca_svd_XV     took: 0.3502 seconds | svd([800,3200]) + [800,3200]*[3200,10]
# pca_svd_US     took: 0.3481 seconds | svd([800,3200]) + [800,10]*[10]
# pca_svd_tX_VhS took: 0.2413 seconds | svd([3200,800]) + [3200,10]*[10]

# Checking that we have the same results.
# mean_absolute_error(pca_svd_US, pca_cov)  = 0.001356658462838996
# mean_absolute_error(pca_svd_US, -pca_cov) = 0.01035609352144603
# ---------------------------------------------
# mean_absolute_error(pca_svd_US, pca_svd_tXU)  = 0.0049716254452656
# mean_absolute_error(pca_svd_US, -pca_svd_tXU) = 0.006741126539017202
# ---------------------------------------------
# mean_absolute_error(pca_svd_US, pca_svd_XV)  = 1.863595858537189e-17
# mean_absolute_error(pca_svd_US, -pca_svd_XV) = 0.01171275198428274
# ---------------------------------------------
# mean_absolute_error(pca_svd_US, pca_svd_tX_VhS)  = 0.004971625445265589
# mean_absolute_error(pca_svd_US, -pca_svd_tX_VhS) = 0.006741126539017209
# ---------------------------------------------

# ###########################
# Starting a new experiment

# Matrix of shape: [1600, 100]
# Target PCA:      [1600, 10]

# Hilbert matrix creation took: 0.0044 seconds
# pca_cov        took: 0.0037 seconds | [100,1600]*[1600,100] + symeig([100,100])
# pca_svd_tXU    took: 0.0430 seconds | svd([100,1600]) + [1600,100]*[100,10]
# pca_svd_XV     took: 0.0079 seconds | svd([1600,100]) + [1600,100]*[100,10]
# pca_svd_US     took: 0.0072 seconds | svd([1600,100]) + [1600,10]*[10]
# pca_svd_tX_VhS took: 0.0493 seconds | svd([100,1600]) + [100,10]*[10]

# Checking that we have the same results.
# mean_absolute_error(pca_svd_US, pca_cov)  = 0.002066378919370285
# mean_absolute_error(pca_svd_US, -pca_cov) = 0.004641379672416099
# ---------------------------------------------
# mean_absolute_error(pca_svd_US, pca_svd_tXU)  = 0.002694527618820402
# mean_absolute_error(pca_svd_US, -pca_svd_tXU) = 0.004013230972961333
# ---------------------------------------------
# mean_absolute_error(pca_svd_US, pca_svd_XV)  = 7.897000931166144e-18
# mean_absolute_error(pca_svd_US, -pca_svd_XV) = 0.006707758591781758
# ---------------------------------------------
# mean_absolute_error(pca_svd_US, pca_svd_tX_VhS)  = 0.002694527618820403
# mean_absolute_error(pca_svd_US, -pca_svd_tX_VhS) = 0.004013230972961332
# ---------------------------------------------

# ###########################
# Starting a new experiment

# Matrix of shape: [1600, 200]
# Target PCA:      [1600, 10]

# Hilbert matrix creation took: 0.0043 seconds
# pca_cov        took: 0.0096 seconds | [200,1600]*[1600,200] + symeig([200,200])
# pca_svd_tXU    took: 0.0712 seconds | svd([200,1600]) + [1600,200]*[200,10]
# pca_svd_XV     took: 0.0376 seconds | svd([1600,200]) + [1600,200]*[200,10]
# pca_svd_US     took: 0.0267 seconds | svd([1600,200]) + [1600,10]*[10]
# pca_svd_tX_VhS took: 0.0459 seconds | svd([200,1600]) + [200,10]*[10]

# Checking that we have the same results.
# mean_absolute_error(pca_svd_US, pca_cov)  = 0.002368933245577393
# mean_absolute_error(pca_svd_US, -pca_cov) = 0.005152123795156442
# ---------------------------------------------
# mean_absolute_error(pca_svd_US, pca_svd_tXU)  = 0.0009006127381817561
# mean_absolute_error(pca_svd_US, -pca_svd_tXU) = 0.006620444302550262
# ---------------------------------------------
# mean_absolute_error(pca_svd_US, pca_svd_XV)  = 9.974110446214904e-18
# mean_absolute_error(pca_svd_US, -pca_svd_XV) = 0.007521057040732003
# ---------------------------------------------
# mean_absolute_error(pca_svd_US, pca_svd_tX_VhS)  = 0.0009006127381817557
# mean_absolute_error(pca_svd_US, -pca_svd_tX_VhS) = 0.006620444302550263
# ---------------------------------------------

# ###########################
# Starting a new experiment

# Matrix of shape: [1600, 400]
# Target PCA:      [1600, 10]

# Hilbert matrix creation took: 0.0044 seconds
# pca_cov        took: 0.0455 seconds | [400,1600]*[1600,400] + symeig([400,400])
# pca_svd_tXU    took: 0.0997 seconds | svd([400,1600]) + [1600,400]*[400,10]
# pca_svd_XV     took: 0.0595 seconds | svd([1600,400]) + [1600,400]*[400,10]
# pca_svd_US     took: 0.0594 seconds | svd([1600,400]) + [1600,10]*[10]
# pca_svd_tX_VhS took: 0.0994 seconds | svd([400,1600]) + [400,10]*[10]

# Checking that we have the same results.
# mean_absolute_error(pca_svd_US, pca_cov)  = 0.00259588160522189
# mean_absolute_error(pca_svd_US, -pca_cov) = 0.005519781671840757
# ---------------------------------------------
# mean_absolute_error(pca_svd_US, pca_svd_tXU)  = 0.001085057412495759
# mean_absolute_error(pca_svd_US, -pca_svd_tXU) = 0.007030605864566003
# ---------------------------------------------
# mean_absolute_error(pca_svd_US, pca_svd_XV)  = 1.622486000355446e-17
# mean_absolute_error(pca_svd_US, -pca_svd_XV) = 0.00811566327706178
# ---------------------------------------------
# mean_absolute_error(pca_svd_US, pca_svd_tX_VhS)  = 0.001085057412495758
# mean_absolute_error(pca_svd_US, -pca_svd_tX_VhS) = 0.007030605864566002
# ---------------------------------------------

# ###########################
# Starting a new experiment

# Matrix of shape: [1600, 800]
# Target PCA:      [1600, 10]

# Hilbert matrix creation took: 0.0042 seconds
# pca_cov        took: 0.0973 seconds | [800,1600]*[1600,800] + symeig([800,800])
# pca_svd_tXU    took: 0.2499 seconds | svd([800,1600]) + [1600,800]*[800,10]
# pca_svd_XV     took: 0.1789 seconds | svd([1600,800]) + [1600,800]*[800,10]
# pca_svd_US     took: 0.1729 seconds | svd([1600,800]) + [1600,10]*[10]
# pca_svd_tX_VhS took: 0.2554 seconds | svd([800,1600]) + [800,10]*[10]

# Checking that we have the same results.
# mean_absolute_error(pca_svd_US, pca_cov)  = 0.002604677527240927
# mean_absolute_error(pca_svd_US, -pca_cov) = 0.005866137149025072
# ---------------------------------------------
# mean_absolute_error(pca_svd_US, pca_svd_tXU)  = 0.001084306894856181
# mean_absolute_error(pca_svd_US, -pca_svd_tXU) = 0.007386507781409327
# ---------------------------------------------
# mean_absolute_error(pca_svd_US, pca_svd_XV)  = 1.14792811005122e-17
# mean_absolute_error(pca_svd_US, -pca_svd_XV) = 0.008470814676265481
# ---------------------------------------------
# mean_absolute_error(pca_svd_US, pca_svd_tX_VhS)  = 0.001084306894856181
# mean_absolute_error(pca_svd_US, -pca_svd_tX_VhS) = 0.00738650778140933
# ---------------------------------------------

# ###########################
# Starting a new experiment

# Matrix of shape: [1600, 1600]
# Target PCA:      [1600, 10]

# Hilbert matrix creation took: 0.0044 seconds
# pca_cov        took: 0.2181 seconds | [1600,1600]*[1600,1600] + symeig([1600,1600])
# pca_svd_tXU    took: 0.5143 seconds | svd([1600,1600]) + [1600,1600]*[1600,10]
# pca_svd_XV     took: 0.5339 seconds | svd([1600,1600]) + [1600,1600]*[1600,10]
# pca_svd_US     took: 0.4931 seconds | svd([1600,1600]) + [1600,10]*[10]
# pca_svd_tX_VhS took: 0.5405 seconds | svd([1600,1600]) + [1600,10]*[10]

# Checking that we have the same results.
# mean_absolute_error(pca_svd_US, pca_cov)  = 0.001131684514799117
# mean_absolute_error(pca_svd_US, -pca_cov) = 0.007504165663471739
# ---------------------------------------------
# mean_absolute_error(pca_svd_US, pca_svd_tXU)  = 1.643514106731121e-17
# mean_absolute_error(pca_svd_US, -pca_svd_tXU) = 0.00863585017827067
# ---------------------------------------------
# mean_absolute_error(pca_svd_US, pca_svd_XV)  = 1.838011199928455e-17
# mean_absolute_error(pca_svd_US, -pca_svd_XV) = 0.008635850178270668
# ---------------------------------------------
# mean_absolute_error(pca_svd_US, pca_svd_tX_VhS)  = 1.676757654717949e-17
# mean_absolute_error(pca_svd_US, -pca_svd_tX_VhS) = 0.008635850178270661
# ---------------------------------------------

# ###########################
# Starting a new experiment

# Matrix of shape: [1600, 3200]
# Target PCA:      [1600, 10]

# Hilbert matrix creation took: 0.0663 seconds
# pca_cov        took: 0.9264 seconds | [3200,1600]*[1600,3200] + symeig([3200,3200])
# pca_svd_tXU    took: 0.8017 seconds | svd([3200,1600]) + [1600,3200]*[3200,10]
# pca_svd_XV     took: 0.9300 seconds | svd([1600,3200]) + [1600,3200]*[3200,10]
# pca_svd_US     took: 1.1002 seconds | svd([1600,3200]) + [1600,10]*[10]
# pca_svd_tX_VhS took: 0.7414 seconds | svd([3200,1600]) + [3200,10]*[10]

# Checking that we have the same results.
# mean_absolute_error(pca_svd_US, pca_cov)  = 0.003949205341181414
# mean_absolute_error(pca_svd_US, -pca_cov) = 0.004744170794965908
# ---------------------------------------------
# mean_absolute_error(pca_svd_US, pca_svd_tXU)  = 0.00130286051830964
# mean_absolute_error(pca_svd_US, -pca_svd_tXU) = 0.007390515617837583
# ---------------------------------------------
# mean_absolute_error(pca_svd_US, pca_svd_XV)  = 2.043781586558689e-17
# mean_absolute_error(pca_svd_US, -pca_svd_XV) = 0.008693376136147247
# ---------------------------------------------
# mean_absolute_error(pca_svd_US, pca_svd_tX_VhS)  = 0.001302860518309638
# mean_absolute_error(pca_svd_US, -pca_svd_tX_VhS) = 0.007390515617837583
# ---------------------------------------------

# ###########################
# Starting a new experiment

# Matrix of shape: [3200, 100]
# Target PCA:      [3200, 10]

# Hilbert matrix creation took: 0.0277 seconds
# pca_cov        took: 0.0070 seconds | [100,3200]*[3200,100] + symeig([100,100])
# pca_svd_tXU    took: 0.0525 seconds | svd([100,3200]) + [3200,100]*[100,10]
# pca_svd_XV     took: 0.0158 seconds | svd([3200,100]) + [3200,100]*[100,10]
# pca_svd_US     took: 0.0147 seconds | svd([3200,100]) + [3200,10]*[10]
# pca_svd_tX_VhS took: 0.0445 seconds | svd([100,3200]) + [100,10]*[10]

# Checking that we have the same results.
# mean_absolute_error(pca_svd_US, pca_cov)  = 0.001354577406511764
# mean_absolute_error(pca_svd_US, -pca_cov) = 0.002948771703913812
# ---------------------------------------------
# mean_absolute_error(pca_svd_US, pca_svd_tXU)  = 0.0004584565433006567
# mean_absolute_error(pca_svd_US, -pca_svd_tXU) = 0.003844892567124832
# ---------------------------------------------
# mean_absolute_error(pca_svd_US, pca_svd_XV)  = 4.242839604695877e-18
# mean_absolute_error(pca_svd_US, -pca_svd_XV) = 0.004303349110425491
# ---------------------------------------------
# mean_absolute_error(pca_svd_US, pca_svd_tX_VhS)  = 0.0004584565433006563
# mean_absolute_error(pca_svd_US, -pca_svd_tX_VhS) = 0.003844892567124832
# ---------------------------------------------

# ###########################
# Starting a new experiment

# Matrix of shape: [3200, 200]
# Target PCA:      [3200, 10]

# Hilbert matrix creation took: 0.0273 seconds
# pca_cov        took: 0.0339 seconds | [200,3200]*[3200,200] + symeig([200,200])
# pca_svd_tXU    took: 0.0670 seconds | svd([200,3200]) + [3200,200]*[200,10]
# pca_svd_XV     took: 0.0368 seconds | svd([3200,200]) + [3200,200]*[200,10]
# pca_svd_US     took: 0.0359 seconds | svd([3200,200]) + [3200,10]*[10]
# pca_svd_tX_VhS took: 0.0679 seconds | svd([200,3200]) + [200,10]*[10]

# Checking that we have the same results.
# mean_absolute_error(pca_svd_US, pca_cov)  = 0.001609718988204654
# mean_absolute_error(pca_svd_US, -pca_cov) = 0.003382668674045825
# ---------------------------------------------
# mean_absolute_error(pca_svd_US, pca_svd_tXU)  = 0.000629127792539151
# mean_absolute_error(pca_svd_US, -pca_svd_tXU) = 0.004363259869711024
# ---------------------------------------------
# mean_absolute_error(pca_svd_US, pca_svd_XV)  = 4.095806059385109e-18
# mean_absolute_error(pca_svd_US, -pca_svd_XV) = 0.004992387662250204
# ---------------------------------------------
# mean_absolute_error(pca_svd_US, pca_svd_tX_VhS)  = 0.0006291277925391512
# mean_absolute_error(pca_svd_US, -pca_svd_tX_VhS) = 0.004363259869711023
# ---------------------------------------------

# ###########################
# Starting a new experiment

# Matrix of shape: [3200, 400]
# Target PCA:      [3200, 10]

# Hilbert matrix creation took: 0.0329 seconds
# pca_cov        took: 0.0620 seconds | [400,3200]*[3200,400] + symeig([400,400])
# pca_svd_tXU    took: 0.1513 seconds | svd([400,3200]) + [3200,400]*[400,10]
# pca_svd_XV     took: 0.0875 seconds | svd([3200,400]) + [3200,400]*[400,10]
# pca_svd_US     took: 0.0794 seconds | svd([3200,400]) + [3200,10]*[10]
# pca_svd_tX_VhS took: 0.1543 seconds | svd([400,3200]) + [400,10]*[10]

# Checking that we have the same results.
# mean_absolute_error(pca_svd_US, pca_cov)  = 0.001833849422542441
# mean_absolute_error(pca_svd_US, -pca_cov) = 0.003746670787913528
# ---------------------------------------------
# mean_absolute_error(pca_svd_US, pca_svd_tXU)  = 0.0007949433126909562
# mean_absolute_error(pca_svd_US, -pca_svd_tXU) = 0.004785576897764567
# ---------------------------------------------
# mean_absolute_error(pca_svd_US, pca_svd_XV)  = 5.468623858156532e-18
# mean_absolute_error(pca_svd_US, -pca_svd_XV) = 0.00558052021045553
# ---------------------------------------------
# mean_absolute_error(pca_svd_US, pca_svd_tX_VhS)  = 0.0007949433126909548
# mean_absolute_error(pca_svd_US, -pca_svd_tX_VhS) = 0.004785576897764566
# ---------------------------------------------

# ###########################
# Starting a new experiment

# Matrix of shape: [3200, 800]
# Target PCA:      [3200, 10]

# Hilbert matrix creation took: 0.0432 seconds
# pca_cov        took: 0.1803 seconds | [800,3200]*[3200,800] + symeig([800,800])
# pca_svd_tXU    took: 0.3956 seconds | svd([800,3200]) + [3200,800]*[800,10]
# pca_svd_XV     took: 0.2740 seconds | svd([3200,800]) + [3200,800]*[800,10]
# pca_svd_US     took: 0.2271 seconds | svd([3200,800]) + [3200,10]*[10]
# pca_svd_tX_VhS took: 0.4726 seconds | svd([800,3200]) + [800,10]*[10]

# Checking that we have the same results.
# mean_absolute_error(pca_svd_US, pca_cov)  = 0.00189604420097131
# mean_absolute_error(pca_svd_US, -pca_cov) = 0.004114551080241789
# ---------------------------------------------
# mean_absolute_error(pca_svd_US, pca_svd_tXU)  = 0.0009301233088147818
# mean_absolute_error(pca_svd_US, -pca_svd_tXU) = 0.005080471972398136
# ---------------------------------------------
# mean_absolute_error(pca_svd_US, pca_svd_XV)  = 6.322924166703976e-18
# mean_absolute_error(pca_svd_US, -pca_svd_XV) = 0.006010595281212887
# ---------------------------------------------
# mean_absolute_error(pca_svd_US, pca_svd_tX_VhS)  = 0.0009301233088147805
# mean_absolute_error(pca_svd_US, -pca_svd_tX_VhS) = 0.005080471972398136
# ---------------------------------------------

# ###########################
# Starting a new experiment

# Matrix of shape: [3200, 1600]
# Target PCA:      [3200, 10]

# Hilbert matrix creation took: 0.0394 seconds
# pca_cov        took: 0.2701 seconds | [1600,3200]*[3200,1600] + symeig([1600,1600])
# pca_svd_tXU    took: 0.9893 seconds | svd([1600,3200]) + [3200,1600]*[1600,10]
# pca_svd_XV     took: 0.7409 seconds | svd([3200,1600]) + [3200,1600]*[1600,10]
# pca_svd_US     took: 0.7341 seconds | svd([3200,1600]) + [3200,10]*[10]
# pca_svd_tX_VhS took: 1.0013 seconds | svd([1600,3200]) + [1600,10]*[10]

# Checking that we have the same results.
# mean_absolute_error(pca_svd_US, pca_cov)  = 0.00196786865454476
# mean_absolute_error(pca_svd_US, -pca_cov) = 0.004298963731573183
# ---------------------------------------------
# mean_absolute_error(pca_svd_US, pca_svd_tXU)  = 0.00102855955585176
# mean_absolute_error(pca_svd_US, -pca_svd_tXU) = 0.005238272830266028
# ---------------------------------------------
# mean_absolute_error(pca_svd_US, pca_svd_XV)  = 6.535510861681479e-18
# mean_absolute_error(pca_svd_US, -pca_svd_XV) = 0.006266832386117771
# ---------------------------------------------
# mean_absolute_error(pca_svd_US, pca_svd_tX_VhS)  = 0.001028559555851758
# mean_absolute_error(pca_svd_US, -pca_svd_tX_VhS) = 0.005238272830266031
# ---------------------------------------------

# ###########################
# Starting a new experiment

# Matrix of shape: [3200, 3200]
# Target PCA:      [3200, 10]

# Hilbert matrix creation took: 0.0276 seconds
# pca_cov        took: 0.8528 seconds | [3200,3200]*[3200,3200] + symeig([3200,3200])
# pca_svd_tXU    took: 3.7662 seconds | svd([3200,3200]) + [3200,3200]*[3200,10]
# pca_svd_XV     took: 3.7247 seconds | svd([3200,3200]) + [3200,3200]*[3200,10]
# pca_svd_US     took: 3.4662 seconds | svd([3200,3200]) + [3200,10]*[10]
# pca_svd_tX_VhS took: 3.8202 seconds | svd([3200,3200]) + [3200,10]*[10]

# Checking that we have the same results.
# mean_absolute_error(pca_svd_US, pca_cov)  = 0.0009401260244585959
# mean_absolute_error(pca_svd_US, -pca_cov) = 0.005445552585551724
# ---------------------------------------------
# mean_absolute_error(pca_svd_US, pca_svd_tXU)  = 1.293432812938107e-17
# mean_absolute_error(pca_svd_US, -pca_svd_tXU) = 0.00638567861001017
# ---------------------------------------------
# mean_absolute_error(pca_svd_US, pca_svd_XV)  = 1.260948448786824e-17
# mean_absolute_error(pca_svd_US, -pca_svd_XV) = 0.00638567861001017
# ---------------------------------------------
# mean_absolute_error(pca_svd_US, pca_svd_tX_VhS)  = 1.267763932055683e-17
# mean_absolute_error(pca_svd_US, -pca_svd_tX_VhS) = 0.006385678610010167
# ---------------------------------------------
