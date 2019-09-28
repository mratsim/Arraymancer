import
  times, strformat,
  ../../src/arraymancer

# Benchmarks of PCA implementations

# Helpers
# ---------------------------------------------------------------------------------
proc diag[T](m, n: int, d: Tensor[T]): Tensor[T] {.noInit.}=
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

  # We special case for SVD U*S multiplication
  # which is in Fortran order (col major)
  #
  # Result in C order (row major)
  let M = x.shape[0]
  assert x.shape[1] == K
  assert x.is_F_contiguous()
  assert diag.is_F_contiguous()

  result = zeros[T](M, N)
  let R = result.dataArray
  let X = x.dataArray
  let D = diag.dataArray

  const Tile = 32
  let nk = min(N, K)

  # TODO parallel
  for i in countup(0, M-1, Tile):
    for j in countup(0, nk-1, Tile):
      for ii in i ..< min(i+Tile, M):
        for jj in j ..< min(j+Tile, nk):
          R[ii*N+jj] = X[jj*M+ii] * D[jj]

# Implementations
# ---------------------------------------------------------------------------------

proc pca_cov[T: SomeFloat](X: Tensor[T], n_components = 2): Tensor[T] {.noInit.}=
  # mean_centered
  let X = X .- X.mean(axis=0)
  let m = X.shape[0]
  let n = X.shape[1]

  # Compute covariance matrix
  var cov_matrix = newTensorUninit[T]([n, n])
  gemm(1.T / T(m-1), X.transpose, X, 0, cov_matrix)

  let (_, eigvecs) = cov_matrix.symeig(true, ^n_components .. ^1)

  let rotation_matrix = eigvecs[_, ^1..0|-1]
  result = X * rotation_matrix

proc pca_svd_tXU[T: SomeFloat](X: Tensor[T], n_components = 2): Tensor[T] {.noInit.}=
  let X = X .- X.mean(axis=0)

  let (U, S, Vh) = svd(X.transpose)
  result = X * U[_, 0..<n_components]

proc pca_svd_US[T: SomeFloat](X: Tensor[T], n_components = 2): Tensor[T] {.noInit.}=
  let X = X .- X.mean(axis=0)

  let (U, S, Vh) = svd(X)

  result = mul_diag(U[_, 0..<n_components], S[0..<n_components], n_components, n_components)

proc pca_svd_XV[T: SomeFloat](X: Tensor[T], n_components = 2): Tensor[T] {.noInit.}=
  let X = X .- X.mean(axis=0)

  let (U, S, Vh) = svd(X)
  result = X * Vh.transpose[_, 0..<n_components]

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
      echo "mean_absolute_error(", xStr, ", ", yStr,") = ", err
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
    echo &"pca_cov took:     {stop-start:>4.4f} seconds | [{n},{m}]*[{m},{n}] + symeig([{n},{n}])"

    profile:
      let pca_svd_tXU = pca_svd_tXU(H, k)
    echo &"pca_svd_tXU took: {stop-start:>4.4f} seconds | svd([{n},{m}]) + [{m},{n}]*[{n},{k}]"

    profile:
      let pca_svd_US = pca_svd_US(H, k)
    echo &"pca_svd_US took:  {stop-start:>4.4f} seconds | svd([{m},{n}]) + [{m},{k}]*[{k}]"

    profile:
      let pca_svd_XV = pca_svd_XV(H, k)
    echo &"pca_svd_XV took:  {stop-start:>4.4f} seconds | svd([{m},{n}]) + [{m},{n}]*[{n},{k}]"

    echo &"\nChecking that we have the same results."
    checkError(pca_cov, pca_svd_tXU)
    checkError(pca_cov, pca_svd_US)
    checkError(pca_cov, pca_svd_XV)

    when Display:
      echo "\n--- pca_cov -------------------"
      echo pca_cov
      echo "\n--- pca_svd_tXU -------------------"
      echo pca_svd_tXU
      echo "\n--- pca_svd_US -------------------"
      echo pca_svd_US
      echo "\n--- pca_svd_XV -------------------"
      echo pca_svd_XV

# ###########################
# Starting a new experiment

# Matrix of shape: [100, 100]
# Target PCA:      [100, 10]

# Hilbert matrix creation took: 0.0001 seconds
# pca_cov took:     0.0048 seconds | [100,100]*[100,100] + symeig([100,100])
# pca_svd_tXU took: 0.0026 seconds | svd([100,100]) + [100,100]*[100,10]
# pca_svd_US took:  0.0022 seconds | svd([100,100]) + [100,10]*[10]
# pca_svd_XV took:  0.0021 seconds | svd([100,100]) + [100,100]*[100,10]

# Checking that we have the same results.
# mean_absolute_error(pca_cov, pca_svd_tXU) = 0.001932655651739894
# mean_absolute_error(pca_cov, -pca_svd_tXU) = 0.02475863673565516
# ---------------------------------------------
# mean_absolute_error(pca_cov, pca_svd_US) = 0.001932655651739896
# mean_absolute_error(pca_cov, -pca_svd_US) = 0.02475863673565516
# ---------------------------------------------
# mean_absolute_error(pca_cov, pca_svd_XV) = 0.001932655651739905
# mean_absolute_error(pca_cov, -pca_svd_XV) = 0.02475863673565516
# ---------------------------------------------

# ###########################
# Starting a new experiment

# Matrix of shape: [100, 200]
# Target PCA:      [100, 10]

# Hilbert matrix creation took: 0.0002 seconds
# pca_cov took:     0.0064 seconds | [200,100]*[100,200] + symeig([200,200])
# pca_svd_tXU took: 0.0043 seconds | svd([200,100]) + [100,200]*[200,10]
# pca_svd_US took:  0.0070 seconds | svd([100,200]) + [100,10]*[10]
# pca_svd_XV took:  0.0063 seconds | svd([100,200]) + [100,200]*[200,10]

# Checking that we have the same results.
# mean_absolute_error(pca_cov, pca_svd_tXU) = 0.006570897873960142
# mean_absolute_error(pca_cov, -pca_svd_tXU) = 0.02033659411180166
# ---------------------------------------------
# mean_absolute_error(pca_cov, pca_svd_US) = 0.002030719975565127
# mean_absolute_error(pca_cov, -pca_svd_US) = 0.0248767720101967
# ---------------------------------------------
# mean_absolute_error(pca_cov, pca_svd_XV) = 0.002030719975565127
# mean_absolute_error(pca_cov, -pca_svd_XV) = 0.0248767720101967
# ---------------------------------------------

# ###########################
# Starting a new experiment

# Matrix of shape: [100, 400]
# Target PCA:      [100, 10]

# Hilbert matrix creation took: 0.0006 seconds
# pca_cov took:     0.0195 seconds | [400,100]*[100,400] + symeig([400,400])
# pca_svd_tXU took: 0.0058 seconds | svd([400,100]) + [100,400]*[400,10]
# pca_svd_US took:  0.0099 seconds | svd([100,400]) + [100,10]*[10]
# pca_svd_XV took:  0.0084 seconds | svd([100,400]) + [100,400]*[400,10]

# Checking that we have the same results.
# mean_absolute_error(pca_cov, pca_svd_tXU) = 0.006593310822730553
# mean_absolute_error(pca_cov, -pca_svd_tXU) = 0.02037116569875735
# ---------------------------------------------
# mean_absolute_error(pca_cov, pca_svd_US) = 0.002059295033802309
# mean_absolute_error(pca_cov, -pca_svd_US) = 0.02490518148768566
# ---------------------------------------------
# mean_absolute_error(pca_cov, pca_svd_XV) = 0.002059295033802312
# mean_absolute_error(pca_cov, -pca_svd_XV) = 0.02490518148768567
# ---------------------------------------------

# ###########################
# Starting a new experiment

# Matrix of shape: [100, 800]
# Target PCA:      [100, 10]

# Hilbert matrix creation took: 0.0025 seconds
# pca_cov took:     0.0738 seconds | [800,100]*[100,800] + symeig([800,800])
# pca_svd_tXU took: 0.0059 seconds | svd([800,100]) + [100,800]*[800,10]
# pca_svd_US took:  0.0192 seconds | svd([100,800]) + [100,10]*[10]
# pca_svd_XV took:  0.0165 seconds | svd([100,800]) + [100,800]*[800,10]

# Checking that we have the same results.
# mean_absolute_error(pca_cov, pca_svd_tXU) = 0.006598181906149172
# mean_absolute_error(pca_cov, -pca_svd_tXU) = 0.02037798022775394
# ---------------------------------------------
# mean_absolute_error(pca_cov, pca_svd_US) = 0.002065366626663964
# mean_absolute_error(pca_cov, -pca_svd_US) = 0.02491079550723915
# ---------------------------------------------
# mean_absolute_error(pca_cov, pca_svd_XV) = 0.002065366626663962
# mean_absolute_error(pca_cov, -pca_svd_XV) = 0.02491079550723916
# ---------------------------------------------

# ###########################
# Starting a new experiment

# Matrix of shape: [100, 1600]
# Target PCA:      [100, 10]

# Hilbert matrix creation took: 0.0115 seconds
# pca_cov took:     0.3029 seconds | [1600,100]*[100,1600] + symeig([1600,1600])
# pca_svd_tXU took: 0.0083 seconds | svd([1600,100]) + [100,1600]*[1600,10]
# pca_svd_US took:  0.0375 seconds | svd([100,1600]) + [100,10]*[10]
# pca_svd_XV took:  0.0277 seconds | svd([100,1600]) + [100,1600]*[1600,10]

# Checking that we have the same results.
# mean_absolute_error(pca_cov, pca_svd_tXU) = 0.006599015429483075
# mean_absolute_error(pca_cov, -pca_svd_tXU) = 0.02037911704756523
# ---------------------------------------------
# mean_absolute_error(pca_cov, pca_svd_US) = 0.002066398351083621
# mean_absolute_error(pca_cov, -pca_svd_US) = 0.02491173412596473
# ---------------------------------------------
# mean_absolute_error(pca_cov, pca_svd_XV) = 0.002066398351083619
# mean_absolute_error(pca_cov, -pca_svd_XV) = 0.02491173412596473
# ---------------------------------------------

# ###########################
# Starting a new experiment

# Matrix of shape: [100, 3200]
# Target PCA:      [100, 10]

# Hilbert matrix creation took: 0.0477 seconds
# pca_cov took:     1.0299 seconds | [3200,100]*[100,3200] + symeig([3200,3200])
# pca_svd_tXU took: 0.0156 seconds | svd([3200,100]) + [100,3200]*[3200,10]
# pca_svd_US took:  0.0460 seconds | svd([100,3200]) + [100,10]*[10]
# pca_svd_XV took:  0.0386 seconds | svd([100,3200]) + [100,3200]*[3200,10]

# Checking that we have the same results.
# mean_absolute_error(pca_cov, pca_svd_tXU) = 0.006599142721329844
# mean_absolute_error(pca_cov, -pca_svd_tXU) = 0.02037928176474993
# ---------------------------------------------
# mean_absolute_error(pca_cov, pca_svd_US) = 0.002066551364076294
# mean_absolute_error(pca_cov, -pca_svd_US) = 0.02491187312200345
# ---------------------------------------------
# mean_absolute_error(pca_cov, pca_svd_XV) = 0.002066551364076291
# mean_absolute_error(pca_cov, -pca_svd_XV) = 0.02491187312200346
# ---------------------------------------------

# ###########################
# Starting a new experiment

# Matrix of shape: [200, 100]
# Target PCA:      [200, 10]

# Hilbert matrix creation took: 0.0001 seconds
# pca_cov took:     0.0016 seconds | [100,200]*[200,100] + symeig([100,100])
# pca_svd_tXU took: 0.0058 seconds | svd([100,200]) + [200,100]*[100,10]
# pca_svd_US took:  0.0041 seconds | svd([200,100]) + [200,10]*[10]
# pca_svd_XV took:  0.0051 seconds | svd([200,100]) + [200,100]*[100,10]

# Checking that we have the same results.
# mean_absolute_error(pca_cov, pca_svd_tXU) = 0.00182185322894013
# mean_absolute_error(pca_cov, -pca_svd_tXU) = 0.01822828086324626
# ---------------------------------------------
# mean_absolute_error(pca_cov, pca_svd_US) = 0.005370049268789392
# mean_absolute_error(pca_cov, -pca_svd_US) = 0.01468008482339707
# ---------------------------------------------
# mean_absolute_error(pca_cov, pca_svd_XV) = 0.005370049268789392
# mean_absolute_error(pca_cov, -pca_svd_XV) = 0.01468008482339708
# ---------------------------------------------

# ###########################
# Starting a new experiment

# Matrix of shape: [200, 200]
# Target PCA:      [200, 10]

# Hilbert matrix creation took: 0.0001 seconds
# pca_cov took:     0.0054 seconds | [200,200]*[200,200] + symeig([200,200])
# pca_svd_tXU took: 0.0090 seconds | svd([200,200]) + [200,200]*[200,10]
# pca_svd_US took:  0.0246 seconds | svd([200,200]) + [200,10]*[10]
# pca_svd_XV took:  0.0101 seconds | svd([200,200]) + [200,200]*[200,10]

# Checking that we have the same results.
# mean_absolute_error(pca_cov, pca_svd_tXU) = 0.002012538895062469
# mean_absolute_error(pca_cov, -pca_svd_tXU) = 0.01847800800226472
# ---------------------------------------------
# mean_absolute_error(pca_cov, pca_svd_US) = 0.002012538895062467
# mean_absolute_error(pca_cov, -pca_svd_US) = 0.01847800800226472
# ---------------------------------------------
# mean_absolute_error(pca_cov, pca_svd_XV) = 0.002012538895062467
# mean_absolute_error(pca_cov, -pca_svd_XV) = 0.01847800800226472
# ---------------------------------------------

# ###########################
# Starting a new experiment

# Matrix of shape: [200, 400]
# Target PCA:      [200, 10]

# Hilbert matrix creation took: 0.0002 seconds
# pca_cov took:     0.0133 seconds | [400,200]*[200,400] + symeig([400,400])
# pca_svd_tXU took: 0.0150 seconds | svd([400,200]) + [200,400]*[400,10]
# pca_svd_US took:  0.0177 seconds | svd([200,400]) + [200,10]*[10]
# pca_svd_XV took:  0.0208 seconds | svd([200,400]) + [200,400]*[400,10]

# Checking that we have the same results.
# mean_absolute_error(pca_cov, pca_svd_tXU) = 0.005609966351268989
# mean_absolute_error(pca_cov, -pca_svd_tXU) = 0.01503513178897382
# ---------------------------------------------
# mean_absolute_error(pca_cov, pca_svd_US) = 0.001655813037479778
# mean_absolute_error(pca_cov, -pca_svd_US) = 0.01898928510276307
# ---------------------------------------------
# mean_absolute_error(pca_cov, pca_svd_XV) = 0.001655813037479783
# mean_absolute_error(pca_cov, -pca_svd_XV) = 0.01898928510276307
# ---------------------------------------------

# ###########################
# Starting a new experiment

# Matrix of shape: [200, 800]
# Target PCA:      [200, 10]

# Hilbert matrix creation took: 0.0010 seconds
# pca_cov took:     0.0722 seconds | [800,200]*[200,800] + symeig([800,800])
# pca_svd_tXU took: 0.0248 seconds | svd([800,200]) + [200,800]*[800,10]
# pca_svd_US took:  0.0327 seconds | svd([200,800]) + [200,10]*[10]
# pca_svd_XV took:  0.0339 seconds | svd([200,800]) + [200,800]*[800,10]

# Checking that we have the same results.
# mean_absolute_error(pca_cov, pca_svd_tXU) = 0.005619494306026132
# mean_absolute_error(pca_cov, -pca_svd_tXU) = 0.01506631465884791
# ---------------------------------------------
# mean_absolute_error(pca_cov, pca_svd_US) = 0.001669777155822621
# mean_absolute_error(pca_cov, -pca_svd_US) = 0.01901603180905142
# ---------------------------------------------
# mean_absolute_error(pca_cov, pca_svd_XV) = 0.001669777155822629
# mean_absolute_error(pca_cov, -pca_svd_XV) = 0.01901603180905143
# ---------------------------------------------

# ###########################
# Starting a new experiment

# Matrix of shape: [200, 1600]
# Target PCA:      [200, 10]

# Hilbert matrix creation took: 0.0046 seconds
# pca_cov took:     0.2697 seconds | [1600,200]*[200,1600] + symeig([1600,1600])
# pca_svd_tXU took: 0.0512 seconds | svd([1600,200]) + [200,1600]*[1600,10]
# pca_svd_US took:  0.0706 seconds | svd([200,1600]) + [200,10]*[10]
# pca_svd_XV took:  0.0682 seconds | svd([200,1600]) + [200,1600]*[1600,10]

# Checking that we have the same results.
# mean_absolute_error(pca_cov, pca_svd_tXU) = 0.005622925577928753
# mean_absolute_error(pca_cov, -pca_svd_tXU) = 0.01507125597044572
# ---------------------------------------------
# mean_absolute_error(pca_cov, pca_svd_US) = 0.001672761361514709
# mean_absolute_error(pca_cov, -pca_svd_US) = 0.01902142018685975
# ---------------------------------------------
# mean_absolute_error(pca_cov, pca_svd_XV) = 0.001672761361514716
# mean_absolute_error(pca_cov, -pca_svd_XV) = 0.01902142018685976
# ---------------------------------------------

# ###########################
# Starting a new experiment

# Matrix of shape: [200, 3200]
# Target PCA:      [200, 10]

# Hilbert matrix creation took: 0.0253 seconds
# pca_cov took:     0.8086 seconds | [3200,200]*[200,3200] + symeig([3200,3200])
# pca_svd_tXU took: 0.0348 seconds | svd([3200,200]) + [200,3200]*[3200,10]
# pca_svd_US took:  0.0729 seconds | svd([200,3200]) + [200,10]*[10]
# pca_svd_XV took:  0.0718 seconds | svd([200,3200]) + [200,3200]*[3200,10]

# Checking that we have the same results.
# mean_absolute_error(pca_cov, pca_svd_tXU) = 0.005623524303021414
# mean_absolute_error(pca_cov, -pca_svd_tXU) = 0.01507207142438875
# ---------------------------------------------
# mean_absolute_error(pca_cov, pca_svd_US) = 0.001673281525307154
# mean_absolute_error(pca_cov, -pca_svd_US) = 0.01902231420210297
# ---------------------------------------------
# mean_absolute_error(pca_cov, pca_svd_XV) = 0.001673281525307154
# mean_absolute_error(pca_cov, -pca_svd_XV) = 0.01902231420210297
# ---------------------------------------------

# ###########################
# Starting a new experiment

# Matrix of shape: [400, 100]
# Target PCA:      [400, 10]

# Hilbert matrix creation took: 0.0002 seconds
# pca_cov took:     0.0018 seconds | [100,400]*[400,100] + symeig([100,100])
# pca_svd_tXU took: 0.0092 seconds | svd([100,400]) + [400,100]*[100,10]
# pca_svd_US took:  0.0054 seconds | svd([400,100]) + [400,10]*[10]
# pca_svd_XV took:  0.0054 seconds | svd([400,100]) + [400,100]*[100,10]

# Checking that we have the same results.
# mean_absolute_error(pca_cov, pca_svd_tXU) = 0.001219225054894705
# mean_absolute_error(pca_cov, -pca_svd_tXU) = 0.01327700819029944
# ---------------------------------------------
# mean_absolute_error(pca_cov, pca_svd_US) = 0.004140240515741197
# mean_absolute_error(pca_cov, -pca_svd_US) = 0.01035599272945293
# ---------------------------------------------
# mean_absolute_error(pca_cov, pca_svd_XV) = 0.004140240515741197
# mean_absolute_error(pca_cov, -pca_svd_XV) = 0.01035599272945293
# ---------------------------------------------

# ###########################
# Starting a new experiment

# Matrix of shape: [400, 200]
# Target PCA:      [400, 10]

# Hilbert matrix creation took: 0.0002 seconds
# pca_cov took:     0.0053 seconds | [200,400]*[400,200] + symeig([200,200])
# pca_svd_tXU took: 0.0432 seconds | svd([200,400]) + [400,200]*[200,10]
# pca_svd_US took:  0.0299 seconds | svd([400,200]) + [400,10]*[10]
# pca_svd_XV took:  0.0174 seconds | svd([400,200]) + [400,200]*[200,10]

# Checking that we have the same results.
# mean_absolute_error(pca_cov, pca_svd_tXU) = 0.001403196435056738
# mean_absolute_error(pca_cov, -pca_svd_tXU) = 0.0137776799636753
# ---------------------------------------------
# mean_absolute_error(pca_cov, pca_svd_US) = 0.00441437668940445
# mean_absolute_error(pca_cov, -pca_svd_US) = 0.01076649970932751
# ---------------------------------------------
# mean_absolute_error(pca_cov, pca_svd_XV) = 0.00441437668940445
# mean_absolute_error(pca_cov, -pca_svd_XV) = 0.01076649970932751
# ---------------------------------------------

# ###########################
# Starting a new experiment

# Matrix of shape: [400, 400]
# Target PCA:      [400, 10]

# Hilbert matrix creation took: 0.0002 seconds
# pca_cov took:     0.0162 seconds | [400,400]*[400,400] + symeig([400,400])
# pca_svd_tXU took: 0.0507 seconds | svd([400,400]) + [400,400]*[400,10]
# pca_svd_US took:  0.0460 seconds | svd([400,400]) + [400,10]*[10]
# pca_svd_XV took:  0.0351 seconds | svd([400,400]) + [400,400]*[400,10]

# Checking that we have the same results.
# mean_absolute_error(pca_cov, pca_svd_tXU) = 0.001500797345135287
# mean_absolute_error(pca_cov, -pca_svd_tXU) = 0.01399845249749339
# ---------------------------------------------
# mean_absolute_error(pca_cov, pca_svd_US) = 0.001500797345135291
# mean_absolute_error(pca_cov, -pca_svd_US) = 0.01399845249749339
# ---------------------------------------------
# mean_absolute_error(pca_cov, pca_svd_XV) = 0.001500797345135288
# mean_absolute_error(pca_cov, -pca_svd_XV) = 0.01399845249749339
# ---------------------------------------------

# ###########################
# Starting a new experiment

# Matrix of shape: [400, 800]
# Target PCA:      [400, 10]

# Hilbert matrix creation took: 0.0120 seconds
# pca_cov took:     0.0916 seconds | [800,400]*[400,800] + symeig([800,800])
# pca_svd_tXU took: 0.0577 seconds | svd([800,400]) + [400,800]*[800,10]
# pca_svd_US took:  0.0853 seconds | svd([400,800]) + [400,10]*[10]
# pca_svd_XV took:  0.0761 seconds | svd([400,800]) + [400,800]*[800,10]

# Checking that we have the same results.
# mean_absolute_error(pca_cov, pca_svd_tXU) = 0.00454847652816162
# mean_absolute_error(pca_cov, -pca_svd_tXU) = 0.01106192723874462
# ---------------------------------------------
# mean_absolute_error(pca_cov, pca_svd_US) = 0.001537648550348399
# mean_absolute_error(pca_cov, -pca_svd_US) = 0.01407275521655781
# ---------------------------------------------
# mean_absolute_error(pca_cov, pca_svd_XV) = 0.001537648550348398
# mean_absolute_error(pca_cov, -pca_svd_XV) = 0.01407275521655782
# ---------------------------------------------

# ###########################
# Starting a new experiment

# Matrix of shape: [400, 1600]
# Target PCA:      [400, 10]

# Hilbert matrix creation took: 0.0044 seconds
# pca_cov took:     0.2152 seconds | [1600,400]*[400,1600] + symeig([1600,1600])
# pca_svd_tXU took: 0.0673 seconds | svd([1600,400]) + [400,1600]*[1600,10]
# pca_svd_US took:  0.1047 seconds | svd([400,1600]) + [400,10]*[10]
# pca_svd_XV took:  0.1018 seconds | svd([400,1600]) + [400,1600]*[1600,10]

# Checking that we have the same results.
# mean_absolute_error(pca_cov, pca_svd_tXU) = 0.00455862932996224
# mean_absolute_error(pca_cov, -pca_svd_tXU) = 0.01108090556091188
# ---------------------------------------------
# mean_absolute_error(pca_cov, pca_svd_US) = 0.001547844608095407
# mean_absolute_error(pca_cov, -pca_svd_US) = 0.01409169028277872
# ---------------------------------------------
# mean_absolute_error(pca_cov, pca_svd_XV) = 0.001547844608095405
# mean_absolute_error(pca_cov, -pca_svd_XV) = 0.01409169028277873
# ---------------------------------------------

# ###########################
# Starting a new experiment

# Matrix of shape: [400, 3200]
# Target PCA:      [400, 10]

# Hilbert matrix creation took: 0.0321 seconds
# pca_cov took:     0.8728 seconds | [3200,400]*[400,3200] + symeig([3200,3200])
# pca_svd_tXU took: 0.0827 seconds | svd([3200,400]) + [400,3200]*[3200,10]
# pca_svd_US took:  0.1518 seconds | svd([400,3200]) + [400,10]*[10]
# pca_svd_XV took:  0.1580 seconds | svd([400,3200]) + [400,3200]*[3200,10]

# Checking that we have the same results.
# mean_absolute_error(pca_cov, pca_svd_tXU) = 0.004560712501525559
# mean_absolute_error(pca_cov, -pca_svd_tXU) = 0.01108479541317033
# ---------------------------------------------
# mean_absolute_error(pca_cov, pca_svd_US) = 0.001550019737760012
# mean_absolute_error(pca_cov, -pca_svd_US) = 0.01409548817693593
# ---------------------------------------------
# mean_absolute_error(pca_cov, pca_svd_XV) = 0.001550019737760011
# mean_absolute_error(pca_cov, -pca_svd_XV) = 0.01409548817693594
# ---------------------------------------------

# ###########################
# Starting a new experiment

# Matrix of shape: [800, 100]
# Target PCA:      [800, 10]

# Hilbert matrix creation took: 0.0010 seconds
# pca_cov took:     0.0025 seconds | [100,800]*[800,100] + symeig([100,100])
# pca_svd_tXU took: 0.0274 seconds | svd([100,800]) + [800,100]*[100,10]
# pca_svd_US took:  0.0058 seconds | svd([800,100]) + [800,10]*[10]
# pca_svd_XV took:  0.0060 seconds | svd([800,100]) + [800,100]*[100,10]

# Checking that we have the same results.
# mean_absolute_error(pca_cov, pca_svd_tXU) = 0.000922000320210218
# mean_absolute_error(pca_cov, -pca_svd_tXU) = 0.009147778431791761
# ---------------------------------------------
# mean_absolute_error(pca_cov, pca_svd_US) = 0.003010258301095772
# mean_absolute_error(pca_cov, -pca_svd_US) = 0.007059520450906202
# ---------------------------------------------
# mean_absolute_error(pca_cov, pca_svd_XV) = 0.003010258301095773
# mean_absolute_error(pca_cov, -pca_svd_XV) = 0.007059520450906201
# ---------------------------------------------

# ###########################
# Starting a new experiment

# Matrix of shape: [800, 200]
# Target PCA:      [800, 10]

# Hilbert matrix creation took: 0.0010 seconds
# pca_cov took:     0.0064 seconds | [200,800]*[800,200] + symeig([200,200])
# pca_svd_tXU took: 0.0453 seconds | svd([200,800]) + [800,200]*[200,10]
# pca_svd_US took:  0.0281 seconds | svd([800,200]) + [800,10]*[10]
# pca_svd_XV took:  0.0219 seconds | svd([800,200]) + [800,200]*[200,10]

# Checking that we have the same results.
# mean_absolute_error(pca_cov, pca_svd_tXU) = 0.001122770528206383
# mean_absolute_error(pca_cov, -pca_svd_tXU) = 0.009768721086172563
# ---------------------------------------------
# mean_absolute_error(pca_cov, pca_svd_US) = 0.003336269680585353
# mean_absolute_error(pca_cov, -pca_svd_US) = 0.007555221933793573
# ---------------------------------------------
# mean_absolute_error(pca_cov, pca_svd_XV) = 0.003336269680585351
# mean_absolute_error(pca_cov, -pca_svd_XV) = 0.007555221933793572
# ---------------------------------------------

# ###########################
# Starting a new experiment

# Matrix of shape: [800, 400]
# Target PCA:      [800, 10]

# Hilbert matrix creation took: 0.0010 seconds
# pca_cov took:     0.0248 seconds | [400,800]*[800,400] + symeig([400,400])
# pca_svd_tXU took: 0.1165 seconds | svd([400,800]) + [800,400]*[400,10]
# pca_svd_US took:  0.0580 seconds | svd([800,400]) + [800,10]*[10]
# pca_svd_XV took:  0.0576 seconds | svd([800,400]) + [800,400]*[400,10]

# Checking that we have the same results.
# mean_absolute_error(pca_cov, pca_svd_tXU) = 0.001262551192843635
# mean_absolute_error(pca_cov, -pca_svd_tXU) = 0.01012032163152244
# ---------------------------------------------
# mean_absolute_error(pca_cov, pca_svd_US) = 0.0034998034813876
# mean_absolute_error(pca_cov, -pca_svd_US) = 0.007883069342978328
# ---------------------------------------------
# mean_absolute_error(pca_cov, pca_svd_XV) = 0.0034998034813876
# mean_absolute_error(pca_cov, -pca_svd_XV) = 0.007883069342978328
# ---------------------------------------------

# ###########################
# Starting a new experiment

# Matrix of shape: [800, 800]
# Target PCA:      [800, 10]

# Hilbert matrix creation took: 0.0009 seconds
# pca_cov took:     0.0714 seconds | [800,800]*[800,800] + symeig([800,800])
# pca_svd_tXU took: 0.1439 seconds | svd([800,800]) + [800,800]*[800,10]
# pca_svd_US took:  0.1293 seconds | svd([800,800]) + [800,10]*[10]
# pca_svd_XV took:  0.1699 seconds | svd([800,800]) + [800,800]*[800,10]

# Checking that we have the same results.
# mean_absolute_error(pca_cov, pca_svd_tXU) = 0.001324183122544305
# mean_absolute_error(pca_cov, -pca_svd_tXU) = 0.0102873384250499
# ---------------------------------------------
# mean_absolute_error(pca_cov, pca_svd_US) = 0.001324183122544304
# mean_absolute_error(pca_cov, -pca_svd_US) = 0.01028733842504989
# ---------------------------------------------
# mean_absolute_error(pca_cov, pca_svd_XV) = 0.001324183122544312
# mean_absolute_error(pca_cov, -pca_svd_XV) = 0.01028733842504989
# ---------------------------------------------

# ###########################
# Starting a new experiment

# Matrix of shape: [800, 1600]
# Target PCA:      [800, 10]

# Hilbert matrix creation took: 0.0049 seconds
# pca_cov took:     0.2097 seconds | [1600,800]*[800,1600] + symeig([1600,1600])
# pca_svd_tXU took: 0.1924 seconds | svd([1600,800]) + [800,1600]*[1600,10]
# pca_svd_US took:  0.2617 seconds | svd([800,1600]) + [800,10]*[10]
# pca_svd_XV took:  0.2609 seconds | svd([800,1600]) + [800,1600]*[1600,10]

# Checking that we have the same results.
# mean_absolute_error(pca_cov, pca_svd_tXU) = 0.00361113118627248
# mean_absolute_error(pca_cov, -pca_svd_tXU) = 0.008080572984267074
# ---------------------------------------------
# mean_absolute_error(pca_cov, pca_svd_US) = 0.001349796772436461
# mean_absolute_error(pca_cov, -pca_svd_US) = 0.01034190739810298
# ---------------------------------------------
# mean_absolute_error(pca_cov, pca_svd_XV) = 0.00134979677243646
# mean_absolute_error(pca_cov, -pca_svd_XV) = 0.01034190739810298
# ---------------------------------------------

# ###########################
# Starting a new experiment

# Matrix of shape: [800, 3200]
# Target PCA:      [800, 10]

# Hilbert matrix creation took: 0.0260 seconds
# pca_cov took:     0.8001 seconds | [3200,800]*[800,3200] + symeig([3200,3200])
# pca_svd_tXU took: 0.2643 seconds | svd([3200,800]) + [800,3200]*[3200,10]
# pca_svd_US took:  0.3701 seconds | svd([800,3200]) + [800,10]*[10]
# pca_svd_XV took:  0.3660 seconds | svd([800,3200]) + [800,3200]*[3200,10]

# Checking that we have the same results.
# mean_absolute_error(pca_cov, pca_svd_tXU) = 0.003618543419768159
# mean_absolute_error(pca_cov, -pca_svd_tXU) = 0.008094208564516961
# ---------------------------------------------
# mean_absolute_error(pca_cov, pca_svd_US) = 0.001356658462838996
# mean_absolute_error(pca_cov, -pca_svd_US) = 0.01035609352144603
# ---------------------------------------------
# mean_absolute_error(pca_cov, pca_svd_XV) = 0.001356658462838999
# mean_absolute_error(pca_cov, -pca_svd_XV) = 0.01035609352144604
# ---------------------------------------------

# ###########################
# Starting a new experiment

# Matrix of shape: [1600, 100]
# Target PCA:      [1600, 10]

# Hilbert matrix creation took: 0.0044 seconds
# pca_cov took:     0.0029 seconds | [100,1600]*[1600,100] + symeig([100,100])
# pca_svd_tXU took: 0.0372 seconds | svd([100,1600]) + [1600,100]*[100,10]
# pca_svd_US took:  0.0085 seconds | svd([1600,100]) + [1600,10]*[10]
# pca_svd_XV took:  0.0078 seconds | svd([1600,100]) + [1600,100]*[100,10]

# Checking that we have the same results.
# mean_absolute_error(pca_cov, pca_svd_tXU) = 0.0006509043733245349
# mean_absolute_error(pca_cov, -pca_svd_tXU) = 0.006056854218461889
# ---------------------------------------------
# mean_absolute_error(pca_cov, pca_svd_US) = 0.002066378919370285
# mean_absolute_error(pca_cov, -pca_svd_US) = 0.004641379672416099
# ---------------------------------------------
# mean_absolute_error(pca_cov, pca_svd_XV) = 0.002066378919370284
# mean_absolute_error(pca_cov, -pca_svd_XV) = 0.0046413796724161
# ---------------------------------------------

# ###########################
# Starting a new experiment

# Matrix of shape: [1600, 200]
# Target PCA:      [1600, 10]

# Hilbert matrix creation took: 0.0043 seconds
# pca_cov took:     0.0069 seconds | [200,1600]*[1600,200] + symeig([200,200])
# pca_svd_tXU took: 0.0682 seconds | svd([200,1600]) + [1600,200]*[200,10]
# pca_svd_US took:  0.0375 seconds | svd([1600,200]) + [1600,10]*[10]
# pca_svd_XV took:  0.0279 seconds | svd([1600,200]) + [1600,200]*[200,10]

# Checking that we have the same results.
# mean_absolute_error(pca_cov, pca_svd_tXU) = 0.00313243630469686
# mean_absolute_error(pca_cov, -pca_svd_tXU) = 0.004388620736036966
# ---------------------------------------------
# mean_absolute_error(pca_cov, pca_svd_US) = 0.002368933245577393
# mean_absolute_error(pca_cov, -pca_svd_US) = 0.005152123795156442
# ---------------------------------------------
# mean_absolute_error(pca_cov, pca_svd_XV) = 0.002368933245577393
# mean_absolute_error(pca_cov, -pca_svd_XV) = 0.005152123795156446
# ---------------------------------------------

# ###########################
# Starting a new experiment

# Matrix of shape: [1600, 400]
# Target PCA:      [1600, 10]

# Hilbert matrix creation took: 0.0044 seconds
# pca_cov took:     0.0330 seconds | [400,1600]*[1600,400] + symeig([400,400])
# pca_svd_tXU took: 0.1580 seconds | svd([400,1600]) + [1600,400]*[400,10]
# pca_svd_US took:  0.0609 seconds | svd([1600,400]) + [1600,10]*[10]
# pca_svd_XV took:  0.0637 seconds | svd([1600,400]) + [1600,400]*[400,10]

# Checking that we have the same results.
# mean_absolute_error(pca_cov, pca_svd_tXU) = 0.003483627187178229
# mean_absolute_error(pca_cov, -pca_svd_tXU) = 0.004632036089884324
# ---------------------------------------------
# mean_absolute_error(pca_cov, pca_svd_US) = 0.00259588160522189
# mean_absolute_error(pca_cov, -pca_svd_US) = 0.005519781671840757
# ---------------------------------------------
# mean_absolute_error(pca_cov, pca_svd_XV) = 0.002595881605221891
# mean_absolute_error(pca_cov, -pca_svd_XV) = 0.005519781671840767
# ---------------------------------------------

# ###########################
# Starting a new experiment

# Matrix of shape: [1600, 800]
# Target PCA:      [1600, 10]

# Hilbert matrix creation took: 0.0043 seconds
# pca_cov took:     0.0777 seconds | [800,1600]*[1600,800] + symeig([800,800])
# pca_svd_tXU took: 0.2899 seconds | svd([800,1600]) + [1600,800]*[800,10]
# pca_svd_US took:  0.1844 seconds | svd([1600,800]) + [1600,10]*[10]
# pca_svd_XV took:  0.1851 seconds | svd([1600,800]) + [1600,800]*[800,10]

# Checking that we have the same results.
# mean_absolute_error(pca_cov, pca_svd_tXU) = 0.003681752052037469
# mean_absolute_error(pca_cov, -pca_svd_tXU) = 0.004789062624228436
# ---------------------------------------------
# mean_absolute_error(pca_cov, pca_svd_US) = 0.002604677527240927
# mean_absolute_error(pca_cov, -pca_svd_US) = 0.005866137149025072
# ---------------------------------------------
# mean_absolute_error(pca_cov, pca_svd_XV) = 0.002604677527240928
# mean_absolute_error(pca_cov, -pca_svd_XV) = 0.00586613714902507
# ---------------------------------------------

# ###########################
# Starting a new experiment

# Matrix of shape: [1600, 1600]
# Target PCA:      [1600, 10]

# Hilbert matrix creation took: 0.0043 seconds
# pca_cov took:     0.2235 seconds | [1600,1600]*[1600,1600] + symeig([1600,1600])
# pca_svd_tXU took: 0.5845 seconds | svd([1600,1600]) + [1600,1600]*[1600,10]
# pca_svd_US took:  0.5223 seconds | svd([1600,1600]) + [1600,10]*[10]
# pca_svd_XV took:  0.5195 seconds | svd([1600,1600]) + [1600,1600]*[1600,10]

# Checking that we have the same results.
# mean_absolute_error(pca_cov, pca_svd_tXU) = 0.001131684514799118
# mean_absolute_error(pca_cov, -pca_svd_tXU) = 0.007504165663471745
# ---------------------------------------------
# mean_absolute_error(pca_cov, pca_svd_US) = 0.001131684514799117
# mean_absolute_error(pca_cov, -pca_svd_US) = 0.007504165663471739
# ---------------------------------------------
# mean_absolute_error(pca_cov, pca_svd_XV) = 0.001131684514799118
# mean_absolute_error(pca_cov, -pca_svd_XV) = 0.007504165663471743
# ---------------------------------------------

# ###########################
# Starting a new experiment

# Matrix of shape: [1600, 3200]
# Target PCA:      [1600, 10]

# Hilbert matrix creation took: 0.0242 seconds
# pca_cov took:     1.1154 seconds | [3200,1600]*[1600,3200] + symeig([3200,3200])
# pca_svd_tXU took: 0.8394 seconds | svd([3200,1600]) + [1600,3200]*[3200,10]
# pca_svd_US took:  1.1056 seconds | svd([1600,3200]) + [1600,10]*[10]
# pca_svd_XV took:  1.2268 seconds | svd([1600,3200]) + [1600,3200]*[3200,10]

# Checking that we have the same results.
# mean_absolute_error(pca_cov, pca_svd_tXU) = 0.002659151118196231
# mean_absolute_error(pca_cov, -pca_svd_tXU) = 0.006034225017951067
# ---------------------------------------------
# mean_absolute_error(pca_cov, pca_svd_US) = 0.003949205341181414
# mean_absolute_error(pca_cov, -pca_svd_US) = 0.004744170794965908
# ---------------------------------------------
# mean_absolute_error(pca_cov, pca_svd_XV) = 0.003949205341181415
# mean_absolute_error(pca_cov, -pca_svd_XV) = 0.004744170794965908
# ---------------------------------------------

# ###########################
# Starting a new experiment

# Matrix of shape: [3200, 100]
# Target PCA:      [3200, 10]

# Hilbert matrix creation took: 0.0246 seconds
# pca_cov took:     0.0059 seconds | [100,3200]*[3200,100] + symeig([100,100])
# pca_svd_tXU took: 0.0511 seconds | svd([100,3200]) + [3200,100]*[100,10]
# pca_svd_US took:  0.0185 seconds | svd([3200,100]) + [3200,10]*[10]
# pca_svd_XV took:  0.0143 seconds | svd([3200,100]) + [3200,100]*[100,10]

# Checking that we have the same results.
# mean_absolute_error(pca_cov, pca_svd_tXU) = 0.001755656787152645
# mean_absolute_error(pca_cov, -pca_svd_tXU) = 0.002547692323273126
# ---------------------------------------------
# mean_absolute_error(pca_cov, pca_svd_US) = 0.001354577406511764
# mean_absolute_error(pca_cov, -pca_svd_US) = 0.002948771703913812
# ---------------------------------------------
# mean_absolute_error(pca_cov, pca_svd_XV) = 0.001354577406511764
# mean_absolute_error(pca_cov, -pca_svd_XV) = 0.002948771703913811
# ---------------------------------------------

# ###########################
# Starting a new experiment

# Matrix of shape: [3200, 200]
# Target PCA:      [3200, 10]

# Hilbert matrix creation took: 0.0195 seconds
# pca_cov took:     0.0108 seconds | [200,3200]*[3200,200] + symeig([200,200])
# pca_svd_tXU took: 0.0787 seconds | svd([200,3200]) + [3200,200]*[200,10]
# pca_svd_US took:  0.0350 seconds | svd([3200,200]) + [3200,10]*[10]
# pca_svd_XV took:  0.0343 seconds | svd([3200,200]) + [3200,200]*[200,10]

# Checking that we have the same results.
# mean_absolute_error(pca_cov, pca_svd_tXU) = 0.002140043044663197
# mean_absolute_error(pca_cov, -pca_svd_tXU) = 0.002852344617587259
# ---------------------------------------------
# mean_absolute_error(pca_cov, pca_svd_US) = 0.001609718988204654
# mean_absolute_error(pca_cov, -pca_svd_US) = 0.003382668674045825
# ---------------------------------------------
# mean_absolute_error(pca_cov, pca_svd_XV) = 0.001609718988204654
# mean_absolute_error(pca_cov, -pca_svd_XV) = 0.003382668674045825
# ---------------------------------------------

# ###########################
# Starting a new experiment

# Matrix of shape: [3200, 400]
# Target PCA:      [3200, 10]

# Hilbert matrix creation took: 0.0196 seconds
# pca_cov took:     0.0274 seconds | [400,3200]*[3200,400] + symeig([400,400])
# pca_svd_tXU took: 0.1598 seconds | svd([400,3200]) + [3200,400]*[400,10]
# pca_svd_US took:  0.0936 seconds | svd([3200,400]) + [3200,10]*[10]
# pca_svd_XV took:  0.0830 seconds | svd([3200,400]) + [3200,400]*[400,10]

# Checking that we have the same results.
# mean_absolute_error(pca_cov, pca_svd_tXU) = 0.0024785049408249
# mean_absolute_error(pca_cov, -pca_svd_tXU) = 0.00310201526963116
# ---------------------------------------------
# mean_absolute_error(pca_cov, pca_svd_US) = 0.001833849422542441
# mean_absolute_error(pca_cov, -pca_svd_US) = 0.003746670787913528
# ---------------------------------------------
# mean_absolute_error(pca_cov, pca_svd_XV) = 0.00183384942254244
# mean_absolute_error(pca_cov, -pca_svd_XV) = 0.003746670787913525
# ---------------------------------------------

# ###########################
# Starting a new experiment

# Matrix of shape: [3200, 800]
# Target PCA:      [3200, 10]

# Hilbert matrix creation took: 0.0260 seconds
# pca_cov took:     0.0857 seconds | [800,3200]*[3200,800] + symeig([800,800])
# pca_svd_tXU took: 0.3944 seconds | svd([800,3200]) + [3200,800]*[800,10]
# pca_svd_US took:  0.2348 seconds | svd([3200,800]) + [3200,10]*[10]
# pca_svd_XV took:  0.2369 seconds | svd([3200,800]) + [3200,800]*[800,10]

# Checking that we have the same results.
# mean_absolute_error(pca_cov, pca_svd_tXU) = 0.002819157373369499
# mean_absolute_error(pca_cov, -pca_svd_tXU) = 0.003191437907843597
# ---------------------------------------------
# mean_absolute_error(pca_cov, pca_svd_US) = 0.00189604420097131
# mean_absolute_error(pca_cov, -pca_svd_US) = 0.004114551080241789
# ---------------------------------------------
# mean_absolute_error(pca_cov, pca_svd_XV) = 0.00189604420097131
# mean_absolute_error(pca_cov, -pca_svd_XV) = 0.004114551080241788
# ---------------------------------------------

# ###########################
# Starting a new experiment

# Matrix of shape: [3200, 1600]
# Target PCA:      [3200, 10]

# Hilbert matrix creation took: 0.0248 seconds
# pca_cov took:     0.2430 seconds | [1600,3200]*[3200,1600] + symeig([1600,1600])
# pca_svd_tXU took: 1.0530 seconds | svd([1600,3200]) + [3200,1600]*[1600,10]
# pca_svd_US took:  0.8305 seconds | svd([3200,1600]) + [3200,10]*[10]
# pca_svd_XV took:  0.7752 seconds | svd([3200,1600]) + [3200,1600]*[1600,10]

# Checking that we have the same results.
# mean_absolute_error(pca_cov, pca_svd_tXU) = 0.002985712544431292
# mean_absolute_error(pca_cov, -pca_svd_tXU) = 0.003281119841686649
# ---------------------------------------------
# mean_absolute_error(pca_cov, pca_svd_US) = 0.00196786865454476
# mean_absolute_error(pca_cov, -pca_svd_US) = 0.004298963731573183
# ---------------------------------------------
# mean_absolute_error(pca_cov, pca_svd_XV) = 0.001967868654544759
# mean_absolute_error(pca_cov, -pca_svd_XV) = 0.004298963731573184
# ---------------------------------------------

# ###########################
# Starting a new experiment

# Matrix of shape: [3200, 3200]
# Target PCA:      [3200, 10]

# Hilbert matrix creation took: 0.0248 seconds
# pca_cov took:     0.9378 seconds | [3200,3200]*[3200,3200] + symeig([3200,3200])
# pca_svd_tXU took: 3.8584 seconds | svd([3200,3200]) + [3200,3200]*[3200,10]
# pca_svd_US took:  5.4301 seconds | svd([3200,3200]) + [3200,10]*[10]
# pca_svd_XV took:  4.7299 seconds | svd([3200,3200]) + [3200,3200]*[3200,10]

# Checking that we have the same results.
# mean_absolute_error(pca_cov, pca_svd_tXU) = 0.0009401260244585951
# mean_absolute_error(pca_cov, -pca_svd_tXU) = 0.005445552585551723
# ---------------------------------------------
# mean_absolute_error(pca_cov, pca_svd_US) = 0.0009401260244585959
# mean_absolute_error(pca_cov, -pca_svd_US) = 0.005445552585551724
# ---------------------------------------------
# mean_absolute_error(pca_cov, pca_svd_XV) = 0.0009401260244585959
# mean_absolute_error(pca_cov, -pca_svd_XV) = 0.005445552585551725
# ---------------------------------------------
