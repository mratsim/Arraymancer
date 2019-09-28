import
  times, strformat,
  ../../src/arraymancer

# Benchmarks of PCA implementations

# Helper
# ---------------------------------------------------------------------------------
proc diag[T](m, n: int, d: Tensor[T]): Tensor[T] {.noInit.}=
  # Creates a rectangular diagonal matrix
  assert d.rank == 1
  result = zeros[T](m, n)

  let k = min(m,n)
  assert d.size == k

  for i in 0 ..< k:
    result[i,i] = d[i]

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

  # matmul * diagonal matrix == matvec
  result = U * S

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
# pca_cov took:     0.0028 seconds | [100,100]*[100,100] + symeig([100,100])
# pca_svd_tXU took: 0.0029 seconds | svd([100,100]) + [100,100]*[100,10]
# pca_svd_US took:  0.0027 seconds | svd([100,100]) + [100,10]*[10]
# pca_svd_XV took:  0.0030 seconds | svd([100,100]) + [100,100]*[100,10]

# Checking that we have the same results.
# mean_absolute_error(pca_cov, pca_svd_tXU) = 0.001932655651739894
# mean_absolute_error(pca_cov, -pca_svd_tXU) = 0.02475863673565516
# ---------------------------------------------
# mean_absolute_error(pca_cov, pca_svd_US) = 7.142731121971616e+244
# mean_absolute_error(pca_cov, -pca_svd_US) = 0.03073937278338286
# ---------------------------------------------
# mean_absolute_error(pca_cov, pca_svd_XV) = 0.001932655651739905
# mean_absolute_error(pca_cov, -pca_svd_XV) = 0.02475863673565516
# ---------------------------------------------

# ###########################
# Starting a new experiment

# Matrix of shape: [100, 200]
# Target PCA:      [100, 10]

# Hilbert matrix creation took: 0.0002 seconds
# pca_cov took:     0.0050 seconds | [200,100]*[100,200] + symeig([200,200])
# pca_svd_tXU took: 0.0043 seconds | svd([200,100]) + [100,200]*[200,10]
# pca_svd_US took:  0.0055 seconds | svd([100,200]) + [100,10]*[10]
# pca_svd_XV took:  0.0063 seconds | svd([100,200]) + [100,200]*[200,10]

# Checking that we have the same results.
# mean_absolute_error(pca_cov, pca_svd_tXU) = 0.006570897873960142
# mean_absolute_error(pca_cov, -pca_svd_tXU) = 0.02033659411180166
# ---------------------------------------------
# mean_absolute_error(pca_cov, pca_svd_US) = 0.04368611467982267
# mean_absolute_error(pca_cov, -pca_svd_US) = 1.428546224394323e+245
# ---------------------------------------------
# mean_absolute_error(pca_cov, pca_svd_XV) = 0.002030719975565127
# mean_absolute_error(pca_cov, -pca_svd_XV) = 0.0248767720101967
# ---------------------------------------------

# ###########################
# Starting a new experiment

# Matrix of shape: [100, 400]
# Target PCA:      [100, 10]

# Hilbert matrix creation took: 0.0007 seconds
# pca_cov took:     0.0145 seconds | [400,100]*[100,400] + symeig([400,400])
# pca_svd_tXU took: 0.0061 seconds | svd([400,100]) + [100,400]*[400,10]
# pca_svd_US took:  0.0087 seconds | svd([100,400]) + [100,10]*[10]
# pca_svd_XV took:  0.0096 seconds | svd([100,400]) + [100,400]*[400,10]

# Checking that we have the same results.
# mean_absolute_error(pca_cov, pca_svd_tXU) = 0.006593310822730553
# mean_absolute_error(pca_cov, -pca_svd_tXU) = 0.02037116569875735
# ---------------------------------------------
# mean_absolute_error(pca_cov, pca_svd_US) = 0.04990665732772084
# mean_absolute_error(pca_cov, -pca_svd_US) = 0.04445582041268038
# ---------------------------------------------
# mean_absolute_error(pca_cov, pca_svd_XV) = 0.002059295033802312
# mean_absolute_error(pca_cov, -pca_svd_XV) = 0.02490518148768567
# ---------------------------------------------

# ###########################
# Starting a new experiment

# Matrix of shape: [100, 800]
# Target PCA:      [100, 10]

# Hilbert matrix creation took: 0.0025 seconds
# pca_cov took:     0.0779 seconds | [800,100]*[100,800] + symeig([800,800])
# pca_svd_tXU took: 0.0058 seconds | svd([800,100]) + [100,800]*[800,10]
# pca_svd_US took:  0.0465 seconds | svd([100,800]) + [100,10]*[10]
# pca_svd_XV took:  0.0174 seconds | svd([100,800]) + [100,800]*[800,10]

# Checking that we have the same results.
# mean_absolute_error(pca_cov, pca_svd_tXU) = 0.006598181906149172
# mean_absolute_error(pca_cov, -pca_svd_tXU) = 0.02037798022775394
# ---------------------------------------------
# mean_absolute_error(pca_cov, pca_svd_US) = 0.07661506030699536
# mean_absolute_error(pca_cov, -pca_svd_US) = 0.02589230900624447
# ---------------------------------------------
# mean_absolute_error(pca_cov, pca_svd_XV) = 0.002065366626663962
# mean_absolute_error(pca_cov, -pca_svd_XV) = 0.02491079550723916
# ---------------------------------------------

# ###########################
# Starting a new experiment

# Matrix of shape: [100, 1600]
# Target PCA:      [100, 10]

# Hilbert matrix creation took: 0.0098 seconds
# pca_cov took:     0.2159 seconds | [1600,100]*[100,1600] + symeig([1600,1600])
# pca_svd_tXU took: 0.0087 seconds | svd([1600,100]) + [100,1600]*[1600,10]
# pca_svd_US took:  0.0429 seconds | svd([100,1600]) + [100,10]*[10]
# pca_svd_XV took:  0.0266 seconds | svd([100,1600]) + [100,1600]*[1600,10]

# Checking that we have the same results.
# mean_absolute_error(pca_cov, pca_svd_tXU) = 0.006599015429483075
# mean_absolute_error(pca_cov, -pca_svd_tXU) = 0.02037911704756523
# ---------------------------------------------
# mean_absolute_error(pca_cov, pca_svd_US) = 0.0362431235421891
# mean_absolute_error(pca_cov, -pca_svd_US) = 0.03503666124311604
# ---------------------------------------------
# mean_absolute_error(pca_cov, pca_svd_XV) = 0.002066398351083619
# mean_absolute_error(pca_cov, -pca_svd_XV) = 0.02491173412596473
# ---------------------------------------------

# ###########################
# Starting a new experiment

# Matrix of shape: [100, 3200]
# Target PCA:      [100, 10]

# Hilbert matrix creation took: 0.0494 seconds
# pca_cov took:     0.8347 seconds | [3200,100]*[100,3200] + symeig([3200,3200])
# pca_svd_tXU took: 0.0149 seconds | svd([3200,100]) + [100,3200]*[3200,10]
# pca_svd_US took:  0.0441 seconds | svd([100,3200]) + [100,10]*[10]
# pca_svd_XV took:  0.0395 seconds | svd([100,3200]) + [100,3200]*[3200,10]

# Checking that we have the same results.
# mean_absolute_error(pca_cov, pca_svd_tXU) = 0.006599142721329844
# mean_absolute_error(pca_cov, -pca_svd_tXU) = 0.02037928176474993
# ---------------------------------------------
# mean_absolute_error(pca_cov, pca_svd_US) = 0.02793870193305361
# mean_absolute_error(pca_cov, -pca_svd_US) = 0.02479559900857717
# ---------------------------------------------
# mean_absolute_error(pca_cov, pca_svd_XV) = 0.002066551364076291
# mean_absolute_error(pca_cov, -pca_svd_XV) = 0.02491187312200346
# ---------------------------------------------

# ###########################
# Starting a new experiment

# Matrix of shape: [200, 100]
# Target PCA:      [200, 10]

# Hilbert matrix creation took: 0.0000 seconds
# pca_cov took:     0.0017 seconds | [100,200]*[200,100] + symeig([100,100])
# pca_svd_tXU took: 0.0054 seconds | svd([100,200]) + [200,100]*[100,10]
# pca_svd_US took:  0.0044 seconds | svd([200,100]) + [200,10]*[10]
# pca_svd_XV took:  0.0042 seconds | svd([200,100]) + [200,100]*[100,10]

# Checking that we have the same results.
# mean_absolute_error(pca_cov, pca_svd_tXU) = 0.00182185322894013
# mean_absolute_error(pca_cov, -pca_svd_tXU) = 0.01822828086324626
# ---------------------------------------------
# mean_absolute_error(pca_cov, pca_svd_US) = 0.03074142356006382
# mean_absolute_error(pca_cov, -pca_svd_US) = 0.02722641819778636
# ---------------------------------------------
# mean_absolute_error(pca_cov, pca_svd_XV) = 0.005370049268789392
# mean_absolute_error(pca_cov, -pca_svd_XV) = 0.01468008482339708
# ---------------------------------------------

# ###########################
# Starting a new experiment

# Matrix of shape: [200, 200]
# Target PCA:      [200, 10]

# Hilbert matrix creation took: 0.0000 seconds
# pca_cov took:     0.0049 seconds | [200,200]*[200,200] + symeig([200,200])
# pca_svd_tXU took: 0.0084 seconds | svd([200,200]) + [200,200]*[200,10]
# pca_svd_US took:  0.0077 seconds | svd([200,200]) + [200,10]*[10]
# pca_svd_XV took:  0.0244 seconds | svd([200,200]) + [200,200]*[200,10]

# Checking that we have the same results.
# mean_absolute_error(pca_cov, pca_svd_tXU) = 0.002012538895062469
# mean_absolute_error(pca_cov, -pca_svd_tXU) = 0.01847800800226472
# ---------------------------------------------
# mean_absolute_error(pca_cov, pca_svd_US) = 0.1572006728646304
# mean_absolute_error(pca_cov, -pca_svd_US) = 0.02514353331912735
# ---------------------------------------------
# mean_absolute_error(pca_cov, pca_svd_XV) = 0.002012538895062467
# mean_absolute_error(pca_cov, -pca_svd_XV) = 0.01847800800226472
# ---------------------------------------------

# ###########################
# Starting a new experiment

# Matrix of shape: [200, 400]
# Target PCA:      [200, 10]

# Hilbert matrix creation took: 0.0002 seconds
# pca_cov took:     0.0154 seconds | [400,200]*[200,400] + symeig([400,400])
# pca_svd_tXU took: 0.0164 seconds | svd([400,200]) + [200,400]*[400,10]
# pca_svd_US took:  0.0174 seconds | svd([200,400]) + [200,10]*[10]
# pca_svd_XV took:  0.0219 seconds | svd([200,400]) + [200,400]*[400,10]

# Checking that we have the same results.
# mean_absolute_error(pca_cov, pca_svd_tXU) = 0.005609966351268989
# mean_absolute_error(pca_cov, -pca_svd_tXU) = 0.01503513178897382
# ---------------------------------------------
# mean_absolute_error(pca_cov, pca_svd_US) = 0.03041297117229052
# mean_absolute_error(pca_cov, -pca_svd_US) = nan
# ---------------------------------------------
# mean_absolute_error(pca_cov, pca_svd_XV) = 0.001655813037479783
# mean_absolute_error(pca_cov, -pca_svd_XV) = 0.01898928510276307
# ---------------------------------------------

# ###########################
# Starting a new experiment

# Matrix of shape: [200, 800]
# Target PCA:      [200, 10]

# Hilbert matrix creation took: 0.0011 seconds
# pca_cov took:     0.0679 seconds | [800,200]*[200,800] + symeig([800,800])
# pca_svd_tXU took: 0.0308 seconds | svd([800,200]) + [200,800]*[800,10]
# pca_svd_US took:  0.0319 seconds | svd([200,800]) + [200,10]*[10]
# pca_svd_XV took:  0.0445 seconds | svd([200,800]) + [200,800]*[800,10]

# Checking that we have the same results.
# mean_absolute_error(pca_cov, pca_svd_tXU) = 0.005619494306026132
# mean_absolute_error(pca_cov, -pca_svd_tXU) = 0.01506631465884791
# ---------------------------------------------
# mean_absolute_error(pca_cov, pca_svd_US) = 5.526048269625573e+149
# mean_absolute_error(pca_cov, -pca_svd_US) = 0.02402261821294654
# ---------------------------------------------
# mean_absolute_error(pca_cov, pca_svd_XV) = 0.001669777155822629
# mean_absolute_error(pca_cov, -pca_svd_XV) = 0.01901603180905143
# ---------------------------------------------

# ###########################
# Starting a new experiment

# Matrix of shape: [200, 1600]
# Target PCA:      [200, 10]

# Hilbert matrix creation took: 0.0440 seconds
# pca_cov took:     0.1841 seconds | [1600,200]*[200,1600] + symeig([1600,1600])
# pca_svd_tXU took: 0.0260 seconds | svd([1600,200]) + [200,1600]*[1600,10]
# pca_svd_US took:  0.0426 seconds | svd([200,1600]) + [200,10]*[10]
# pca_svd_XV took:  0.0430 seconds | svd([200,1600]) + [200,1600]*[1600,10]

# Checking that we have the same results.
# mean_absolute_error(pca_cov, pca_svd_tXU) = 0.005622925577928753
# mean_absolute_error(pca_cov, -pca_svd_tXU) = 0.01507125597044572
# ---------------------------------------------
# mean_absolute_error(pca_cov, pca_svd_US) = 0.05975673736392774
# mean_absolute_error(pca_cov, -pca_svd_US) = 2.984463904994791e+263
# ---------------------------------------------
# mean_absolute_error(pca_cov, pca_svd_XV) = 0.001672761361514716
# mean_absolute_error(pca_cov, -pca_svd_XV) = 0.01902142018685976
# ---------------------------------------------

# ###########################
# Starting a new experiment

# Matrix of shape: [200, 3200]
# Target PCA:      [200, 10]

# Hilbert matrix creation took: 0.0218 seconds
# pca_cov took:     1.0493 seconds | [3200,200]*[200,3200] + symeig([3200,3200])
# pca_svd_tXU took: 0.0349 seconds | svd([3200,200]) + [200,3200]*[3200,10]
# pca_svd_US took:  0.0633 seconds | svd([200,3200]) + [200,10]*[10]
# pca_svd_XV took:  0.0666 seconds | svd([200,3200]) + [200,3200]*[3200,10]

# Checking that we have the same results.
# mean_absolute_error(pca_cov, pca_svd_tXU) = 0.005623524303021414
# mean_absolute_error(pca_cov, -pca_svd_tXU) = 0.01507207142438875
# ---------------------------------------------
# mean_absolute_error(pca_cov, pca_svd_US) = 0.01849782428783138
# mean_absolute_error(pca_cov, -pca_svd_US) = 0.03689481110338276
# ---------------------------------------------
# mean_absolute_error(pca_cov, pca_svd_XV) = 0.001673281525307154
# mean_absolute_error(pca_cov, -pca_svd_XV) = 0.01902231420210297
# ---------------------------------------------

# ###########################
# Starting a new experiment

# Matrix of shape: [400, 100]
# Target PCA:      [400, 10]

# Hilbert matrix creation took: 0.0003 seconds
# pca_cov took:     0.0018 seconds | [100,400]*[400,100] + symeig([100,100])
# pca_svd_tXU took: 0.0088 seconds | svd([100,400]) + [400,100]*[100,10]
# pca_svd_US took:  0.0055 seconds | svd([400,100]) + [400,10]*[10]
# pca_svd_XV took:  0.0051 seconds | svd([400,100]) + [400,100]*[100,10]

# Checking that we have the same results.
# mean_absolute_error(pca_cov, pca_svd_tXU) = 0.001219225054894705
# mean_absolute_error(pca_cov, -pca_svd_tXU) = 0.01327700819029944
# ---------------------------------------------
# mean_absolute_error(pca_cov, pca_svd_US) = 7.142731121971616e+244
# mean_absolute_error(pca_cov, -pca_svd_US) = 2.763024134812786e+149
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
# pca_svd_tXU took: 0.0387 seconds | svd([200,400]) + [400,200]*[200,10]
# pca_svd_US took:  0.0154 seconds | svd([400,200]) + [400,10]*[10]
# pca_svd_XV took:  0.0161 seconds | svd([400,200]) + [400,200]*[200,10]

# Checking that we have the same results.
# mean_absolute_error(pca_cov, pca_svd_tXU) = 0.001403196435056738
# mean_absolute_error(pca_cov, -pca_svd_tXU) = 0.0137776799636753
# ---------------------------------------------
# mean_absolute_error(pca_cov, pca_svd_US) = 0.03836993719495122
# mean_absolute_error(pca_cov, -pca_svd_US) = 0.04313091939499961
# ---------------------------------------------
# mean_absolute_error(pca_cov, pca_svd_XV) = 0.00441437668940445
# mean_absolute_error(pca_cov, -pca_svd_XV) = 0.01076649970932751
# ---------------------------------------------

# ###########################
# Starting a new experiment

# Matrix of shape: [400, 400]
# Target PCA:      [400, 10]

# Hilbert matrix creation took: 0.0002 seconds
# pca_cov took:     0.0127 seconds | [400,400]*[400,400] + symeig([400,400])
# pca_svd_tXU took: 0.0331 seconds | svd([400,400]) + [400,400]*[400,10]
# pca_svd_US took:  0.0275 seconds | svd([400,400]) + [400,10]*[10]
# pca_svd_XV took:  0.0279 seconds | svd([400,400]) + [400,400]*[400,10]

# Checking that we have the same results.
# mean_absolute_error(pca_cov, pca_svd_tXU) = 0.001500797345135287
# mean_absolute_error(pca_cov, -pca_svd_tXU) = 0.01399845249749339
# ---------------------------------------------
# mean_absolute_error(pca_cov, pca_svd_US) = 40.37976140252792
# mean_absolute_error(pca_cov, -pca_svd_US) = 0.01570206501911256
# ---------------------------------------------
# mean_absolute_error(pca_cov, pca_svd_XV) = 0.001500797345135288
# mean_absolute_error(pca_cov, -pca_svd_XV) = 0.01399845249749339
# ---------------------------------------------

# ###########################
# Starting a new experiment

# Matrix of shape: [400, 800]
# Target PCA:      [400, 10]

# Hilbert matrix creation took: 0.0008 seconds
# pca_cov took:     0.0754 seconds | [800,400]*[400,800] + symeig([800,800])
# pca_svd_tXU took: 0.0622 seconds | svd([800,400]) + [400,800]*[800,10]
# pca_svd_US took:  0.0612 seconds | svd([400,800]) + [400,10]*[10]
# pca_svd_XV took:  0.0725 seconds | svd([400,800]) + [400,800]*[800,10]

# Checking that we have the same results.
# mean_absolute_error(pca_cov, pca_svd_tXU) = 0.00454847652816162
# mean_absolute_error(pca_cov, -pca_svd_tXU) = 0.01106192723874462
# ---------------------------------------------
# mean_absolute_error(pca_cov, pca_svd_US) = 0.02694166276621968
# mean_absolute_error(pca_cov, -pca_svd_US) = 0.1892542666897312
# ---------------------------------------------
# mean_absolute_error(pca_cov, pca_svd_XV) = 0.001537648550348398
# mean_absolute_error(pca_cov, -pca_svd_XV) = 0.01407275521655782
# ---------------------------------------------

# ###########################
# Starting a new experiment

# Matrix of shape: [400, 1600]
# Target PCA:      [400, 10]

# Hilbert matrix creation took: 0.0044 seconds
# pca_cov took:     0.2041 seconds | [1600,400]*[400,1600] + symeig([1600,1600])
# pca_svd_tXU took: 0.0720 seconds | svd([1600,400]) + [400,1600]*[1600,10]
# pca_svd_US took:  0.0975 seconds | svd([400,1600]) + [400,10]*[10]
# pca_svd_XV took:  0.1384 seconds | svd([400,1600]) + [400,1600]*[1600,10]

# Checking that we have the same results.
# mean_absolute_error(pca_cov, pca_svd_tXU) = 0.00455862932996224
# mean_absolute_error(pca_cov, -pca_svd_tXU) = 0.01108090556091188
# ---------------------------------------------
# mean_absolute_error(pca_cov, pca_svd_US) = 0.01849015810579711
# mean_absolute_error(pca_cov, -pca_svd_US) = 1.950876642321448e+224
# ---------------------------------------------
# mean_absolute_error(pca_cov, pca_svd_XV) = 0.001547844608095405
# mean_absolute_error(pca_cov, -pca_svd_XV) = 0.01409169028277873
# ---------------------------------------------

# ###########################
# Starting a new experiment

# Matrix of shape: [400, 3200]
# Target PCA:      [400, 10]

# Hilbert matrix creation took: 0.0312 seconds
# pca_cov took:     0.8235 seconds | [3200,400]*[400,3200] + symeig([3200,3200])
# pca_svd_tXU took: 0.0771 seconds | svd([3200,400]) + [400,3200]*[3200,10]
# pca_svd_US took:  0.1421 seconds | svd([400,3200]) + [400,10]*[10]
# pca_svd_XV took:  0.1384 seconds | svd([400,3200]) + [400,3200]*[3200,10]

# Checking that we have the same results.
# mean_absolute_error(pca_cov, pca_svd_tXU) = 0.004560712501525559
# mean_absolute_error(pca_cov, -pca_svd_tXU) = 0.01108479541317033
# ---------------------------------------------
# mean_absolute_error(pca_cov, pca_svd_US) = 7.142731121971616e+244
# mean_absolute_error(pca_cov, -pca_svd_US) = 0.01506447542382082
# ---------------------------------------------
# mean_absolute_error(pca_cov, pca_svd_XV) = 0.001550019737760011
# mean_absolute_error(pca_cov, -pca_svd_XV) = 0.01409548817693594
# ---------------------------------------------

# ###########################
# Starting a new experiment

# Matrix of shape: [800, 100]
# Target PCA:      [800, 10]

# Hilbert matrix creation took: 0.0010 seconds
# pca_cov took:     0.0023 seconds | [100,800]*[800,100] + symeig([100,100])
# pca_svd_tXU took: 0.0342 seconds | svd([100,800]) + [800,100]*[100,10]
# pca_svd_US took:  0.0066 seconds | svd([800,100]) + [800,10]*[10]
# pca_svd_XV took:  0.0059 seconds | svd([800,100]) + [800,100]*[100,10]

# Checking that we have the same results.
# mean_absolute_error(pca_cov, pca_svd_tXU) = 0.000922000320210218
# mean_absolute_error(pca_cov, -pca_svd_tXU) = 0.009147778431791761
# ---------------------------------------------
# mean_absolute_error(pca_cov, pca_svd_US) = nan
# mean_absolute_error(pca_cov, -pca_svd_US) = 1.339648065363775e+149
# ---------------------------------------------
# mean_absolute_error(pca_cov, pca_svd_XV) = 0.003010258301095773
# mean_absolute_error(pca_cov, -pca_svd_XV) = 0.007059520450906201
# ---------------------------------------------

# ###########################
# Starting a new experiment

# Matrix of shape: [800, 200]
# Target PCA:      [800, 10]

# Hilbert matrix creation took: 0.0010 seconds
# pca_cov took:     0.0058 seconds | [200,800]*[800,200] + symeig([200,200])
# pca_svd_tXU took: 0.0470 seconds | svd([200,800]) + [800,200]*[200,10]
# pca_svd_US took:  0.0180 seconds | svd([800,200]) + [800,10]*[10]
# pca_svd_XV took:  0.0195 seconds | svd([800,200]) + [800,200]*[200,10]

# Checking that we have the same results.
# mean_absolute_error(pca_cov, pca_svd_tXU) = 0.001122770528206383
# mean_absolute_error(pca_cov, -pca_svd_tXU) = 0.009768721086172563
# ---------------------------------------------
# mean_absolute_error(pca_cov, pca_svd_US) = 0.1142935377906698
# mean_absolute_error(pca_cov, -pca_svd_US) = 0.0182788337107443
# ---------------------------------------------
# mean_absolute_error(pca_cov, pca_svd_XV) = 0.003336269680585351
# mean_absolute_error(pca_cov, -pca_svd_XV) = 0.007555221933793572
# ---------------------------------------------

# ###########################
# Starting a new experiment

# Matrix of shape: [800, 400]
# Target PCA:      [800, 10]

# Hilbert matrix creation took: 0.0010 seconds
# pca_cov took:     0.0319 seconds | [400,800]*[800,400] + symeig([400,400])
# pca_svd_tXU took: 0.1093 seconds | svd([400,800]) + [800,400]*[400,10]
# pca_svd_US took:  0.0516 seconds | svd([800,400]) + [800,10]*[10]
# pca_svd_XV took:  0.0519 seconds | svd([800,400]) + [800,400]*[400,10]

# Checking that we have the same results.
# mean_absolute_error(pca_cov, pca_svd_tXU) = 0.001262551192843635
# mean_absolute_error(pca_cov, -pca_svd_tXU) = 0.01012032163152244
# ---------------------------------------------
# mean_absolute_error(pca_cov, pca_svd_US) = 0.01811629811936798
# mean_absolute_error(pca_cov, -pca_svd_US) = 20.20936227476934
# ---------------------------------------------
# mean_absolute_error(pca_cov, pca_svd_XV) = 0.0034998034813876
# mean_absolute_error(pca_cov, -pca_svd_XV) = 0.007883069342978328
# ---------------------------------------------

# ###########################
# Starting a new experiment

# Matrix of shape: [800, 800]
# Target PCA:      [800, 10]

# Hilbert matrix creation took: 0.0010 seconds
# pca_cov took:     0.0845 seconds | [800,800]*[800,800] + symeig([800,800])
# pca_svd_tXU took: 0.1422 seconds | svd([800,800]) + [800,800]*[800,10]
# pca_svd_US took:  0.1288 seconds | svd([800,800]) + [800,10]*[10]
# pca_svd_XV took:  0.1310 seconds | svd([800,800]) + [800,800]*[800,10]

# Checking that we have the same results.
# mean_absolute_error(pca_cov, pca_svd_tXU) = 0.001324183122544305
# mean_absolute_error(pca_cov, -pca_svd_tXU) = 0.0102873384250499
# ---------------------------------------------
# mean_absolute_error(pca_cov, pca_svd_US) = 0.07338925068062113
# mean_absolute_error(pca_cov, -pca_svd_US) = 20.21227165053788
# ---------------------------------------------
# mean_absolute_error(pca_cov, pca_svd_XV) = 0.001324183122544312
# mean_absolute_error(pca_cov, -pca_svd_XV) = 0.01028733842504989
# ---------------------------------------------

# ###########################
# Starting a new experiment

# Matrix of shape: [800, 1600]
# Target PCA:      [800, 10]

# Hilbert matrix creation took: 0.0045 seconds
# pca_cov took:     0.2210 seconds | [1600,800]*[800,1600] + symeig([1600,1600])
# pca_svd_tXU took: 0.1973 seconds | svd([1600,800]) + [800,1600]*[1600,10]
# pca_svd_US took:  0.2428 seconds | svd([800,1600]) + [800,10]*[10]
# pca_svd_XV took:  0.2836 seconds | svd([800,1600]) + [800,1600]*[1600,10]

# Checking that we have the same results.
# mean_absolute_error(pca_cov, pca_svd_tXU) = 0.00361113118627248
# mean_absolute_error(pca_cov, -pca_svd_tXU) = 0.008080572984267074
# ---------------------------------------------
# mean_absolute_error(pca_cov, pca_svd_US) = 1.255920061278539e+149
# mean_absolute_error(pca_cov, -pca_svd_US) = nan
# ---------------------------------------------
# mean_absolute_error(pca_cov, pca_svd_XV) = 0.00134979677243646
# mean_absolute_error(pca_cov, -pca_svd_XV) = 0.01034190739810298
# ---------------------------------------------

# ###########################
# Starting a new experiment

# Matrix of shape: [800, 3200]
# Target PCA:      [800, 10]

# Hilbert matrix creation took: 0.0257 seconds
# pca_cov took:     0.8419 seconds | [3200,800]*[800,3200] + symeig([3200,3200])
# pca_svd_tXU took: 0.2347 seconds | svd([3200,800]) + [800,3200]*[3200,10]
# pca_svd_US took:  0.3389 seconds | svd([800,3200]) + [800,10]*[10]
# pca_svd_XV took:  0.3768 seconds | svd([800,3200]) + [800,3200]*[3200,10]

# Checking that we have the same results.
# mean_absolute_error(pca_cov, pca_svd_tXU) = 0.003618543419768159
# mean_absolute_error(pca_cov, -pca_svd_tXU) = 0.008094208564516961
# ---------------------------------------------
# mean_absolute_error(pca_cov, pca_svd_US) = 2.481372264398521e+245
# mean_absolute_error(pca_cov, -pca_svd_US) = nan
# ---------------------------------------------
# mean_absolute_error(pca_cov, pca_svd_XV) = 0.001356658462838999
# mean_absolute_error(pca_cov, -pca_svd_XV) = 0.01035609352144604
# ---------------------------------------------

# ###########################
# Starting a new experiment

# Matrix of shape: [1600, 100]
# Target PCA:      [1600, 10]

# Hilbert matrix creation took: 0.0043 seconds
# pca_cov took:     0.0032 seconds | [100,1600]*[1600,100] + symeig([100,100])
# pca_svd_tXU took: 0.0441 seconds | svd([100,1600]) + [1600,100]*[100,10]
# pca_svd_US took:  0.0073 seconds | svd([1600,100]) + [1600,10]*[10]
# pca_svd_XV took:  0.0078 seconds | svd([1600,100]) + [1600,100]*[100,10]

# Checking that we have the same results.
# mean_absolute_error(pca_cov, pca_svd_tXU) = 0.0006509043733245349
# mean_absolute_error(pca_cov, -pca_svd_tXU) = 0.006056854218461889
# ---------------------------------------------
# mean_absolute_error(pca_cov, pca_svd_US) = 0.01326342817651044
# mean_absolute_error(pca_cov, -pca_svd_US) = 0.01688853827191987
# ---------------------------------------------
# mean_absolute_error(pca_cov, pca_svd_XV) = 0.002066378919370284
# mean_absolute_error(pca_cov, -pca_svd_XV) = 0.0046413796724161
# ---------------------------------------------

# ###########################
# Starting a new experiment

# Matrix of shape: [1600, 200]
# Target PCA:      [1600, 10]

# Hilbert matrix creation took: 0.0042 seconds
# pca_cov took:     0.0070 seconds | [200,1600]*[1600,200] + symeig([200,200])
# pca_svd_tXU took: 0.0660 seconds | svd([200,1600]) + [1600,200]*[200,10]
# pca_svd_US took:  0.0419 seconds | svd([1600,200]) + [1600,10]*[10]
# pca_svd_XV took:  0.0264 seconds | svd([1600,200]) + [1600,200]*[200,10]

# Checking that we have the same results.
# mean_absolute_error(pca_cov, pca_svd_tXU) = 0.00313243630469686
# mean_absolute_error(pca_cov, -pca_svd_tXU) = 0.004388620736036966
# ---------------------------------------------
# mean_absolute_error(pca_cov, pca_svd_US) = 1.785682780492904e+244
# mean_absolute_error(pca_cov, -pca_svd_US) = 0.01283184530802373
# ---------------------------------------------
# mean_absolute_error(pca_cov, pca_svd_XV) = 0.002368933245577393
# mean_absolute_error(pca_cov, -pca_svd_XV) = 0.005152123795156446
# ---------------------------------------------

# ###########################
# Starting a new experiment

# Matrix of shape: [1600, 400]
# Target PCA:      [1600, 10]

# Hilbert matrix creation took: 0.0042 seconds
# pca_cov took:     0.0362 seconds | [400,1600]*[1600,400] + symeig([400,400])
# pca_svd_tXU took: 0.1096 seconds | svd([400,1600]) + [1600,400]*[400,10]
# pca_svd_US took:  0.0611 seconds | svd([1600,400]) + [1600,10]*[10]
# pca_svd_XV took:  0.0634 seconds | svd([1600,400]) + [1600,400]*[400,10]

# Checking that we have the same results.
# mean_absolute_error(pca_cov, pca_svd_tXU) = 0.003483627187178229
# mean_absolute_error(pca_cov, -pca_svd_tXU) = 0.004632036089884324
# ---------------------------------------------
# mean_absolute_error(pca_cov, pca_svd_US) = 0.01626429849736135
# mean_absolute_error(pca_cov, -pca_svd_US) = 0.007096271267972311
# ---------------------------------------------
# mean_absolute_error(pca_cov, pca_svd_XV) = 0.002595881605221891
# mean_absolute_error(pca_cov, -pca_svd_XV) = 0.005519781671840767
# ---------------------------------------------

# ###########################
# Starting a new experiment

# Matrix of shape: [1600, 800]
# Target PCA:      [1600, 10]

# Hilbert matrix creation took: 0.0041 seconds
# pca_cov took:     0.0812 seconds | [800,1600]*[1600,800] + symeig([800,800])
# pca_svd_tXU took: 0.2610 seconds | svd([800,1600]) + [1600,800]*[800,10]
# pca_svd_US took:  0.1812 seconds | svd([1600,800]) + [1600,10]*[10]
# pca_svd_XV took:  0.1745 seconds | svd([1600,800]) + [1600,800]*[800,10]

# Checking that we have the same results.
# mean_absolute_error(pca_cov, pca_svd_tXU) = 0.003681752052037469
# mean_absolute_error(pca_cov, -pca_svd_tXU) = 0.004789062624228436
# ---------------------------------------------
# mean_absolute_error(pca_cov, pca_svd_US) = 0.1042551874870833
# mean_absolute_error(pca_cov, -pca_svd_US) = 0.01318783548743937
# ---------------------------------------------
# mean_absolute_error(pca_cov, pca_svd_XV) = 0.002604677527240928
# mean_absolute_error(pca_cov, -pca_svd_XV) = 0.00586613714902507
# ---------------------------------------------

# ###########################
# Starting a new experiment

# Matrix of shape: [1600, 1600]
# Target PCA:      [1600, 10]

# Hilbert matrix creation took: 0.0043 seconds
# pca_cov took:     0.2576 seconds | [1600,1600]*[1600,1600] + symeig([1600,1600])
# pca_svd_tXU took: 0.5077 seconds | svd([1600,1600]) + [1600,1600]*[1600,10]
# pca_svd_US took:  0.4791 seconds | svd([1600,1600]) + [1600,10]*[10]
# pca_svd_XV took:  0.5190 seconds | svd([1600,1600]) + [1600,1600]*[1600,10]

# Checking that we have the same results.
# mean_absolute_error(pca_cov, pca_svd_tXU) = 0.001131684514799118
# mean_absolute_error(pca_cov, -pca_svd_tXU) = 0.007504165663471745
# ---------------------------------------------
# mean_absolute_error(pca_cov, pca_svd_US) = 0.146972724518411
# mean_absolute_error(pca_cov, -pca_svd_US) = 0.01450414251552973
# ---------------------------------------------
# mean_absolute_error(pca_cov, pca_svd_XV) = 0.001131684514799118
# mean_absolute_error(pca_cov, -pca_svd_XV) = 0.007504165663471743
# ---------------------------------------------

# ###########################
# Starting a new experiment

# Matrix of shape: [1600, 3200]
# Target PCA:      [1600, 10]

# Hilbert matrix creation took: 0.0269 seconds
# pca_cov took:     0.8116 seconds | [3200,1600]*[1600,3200] + symeig([3200,3200])
# pca_svd_tXU took: 0.7672 seconds | svd([3200,1600]) + [1600,3200]*[3200,10]
# pca_svd_US took:  0.9407 seconds | svd([1600,3200]) + [1600,10]*[10]
# pca_svd_XV took:  1.1873 seconds | svd([1600,3200]) + [1600,3200]*[3200,10]

# Checking that we have the same results.
# mean_absolute_error(pca_cov, pca_svd_tXU) = 0.002659151118196231
# mean_absolute_error(pca_cov, -pca_svd_tXU) = 0.006034225017951067
# ---------------------------------------------
# mean_absolute_error(pca_cov, pca_svd_US) = 0.01331969127603753
# mean_absolute_error(pca_cov, -pca_svd_US) = 8.92841390246452e+243
# ---------------------------------------------
# mean_absolute_error(pca_cov, pca_svd_XV) = 0.003949205341181415
# mean_absolute_error(pca_cov, -pca_svd_XV) = 0.004744170794965908
# ---------------------------------------------

# ###########################
# Starting a new experiment

# Matrix of shape: [3200, 100]
# Target PCA:      [3200, 10]

# Hilbert matrix creation took: 0.0490 seconds
# pca_cov took:     0.0049 seconds | [100,3200]*[3200,100] + symeig([100,100])
# pca_svd_tXU took: 0.0546 seconds | svd([100,3200]) + [3200,100]*[100,10]
# pca_svd_US took:  0.0148 seconds | svd([3200,100]) + [3200,10]*[10]
# pca_svd_XV took:  0.0164 seconds | svd([3200,100]) + [3200,100]*[100,10]

# Checking that we have the same results.
# mean_absolute_error(pca_cov, pca_svd_tXU) = 0.001755656787152645
# mean_absolute_error(pca_cov, -pca_svd_tXU) = 0.002547692323273126
# ---------------------------------------------
# mean_absolute_error(pca_cov, pca_svd_US) = 0.03497577970944228
# mean_absolute_error(pca_cov, -pca_svd_US) = nan
# ---------------------------------------------
# mean_absolute_error(pca_cov, pca_svd_XV) = 0.001354577406511764
# mean_absolute_error(pca_cov, -pca_svd_XV) = 0.002948771703913811
# ---------------------------------------------

# ###########################
# Starting a new experiment

# Matrix of shape: [3200, 200]
# Target PCA:      [3200, 10]

# Hilbert matrix creation took: 0.0487 seconds
# pca_cov took:     0.0226 seconds | [200,3200]*[3200,200] + symeig([200,200])
# pca_svd_tXU took: 0.0702 seconds | svd([200,3200]) + [3200,200]*[200,10]
# pca_svd_US took:  0.0343 seconds | svd([3200,200]) + [3200,10]*[10]
# pca_svd_XV took:  0.0350 seconds | svd([3200,200]) + [3200,200]*[200,10]

# Checking that we have the same results.
# mean_absolute_error(pca_cov, pca_svd_tXU) = 0.002140043044663197
# mean_absolute_error(pca_cov, -pca_svd_tXU) = 0.002852344617587259
# ---------------------------------------------
# mean_absolute_error(pca_cov, pca_svd_US) = 2.678524170739356e+244
# mean_absolute_error(pca_cov, -pca_svd_US) = nan
# ---------------------------------------------
# mean_absolute_error(pca_cov, pca_svd_XV) = 0.001609718988204654
# mean_absolute_error(pca_cov, -pca_svd_XV) = 0.003382668674045825
# ---------------------------------------------

# ###########################
# Starting a new experiment

# Matrix of shape: [3200, 400]
# Target PCA:      [3200, 10]

# Hilbert matrix creation took: 0.0257 seconds
# pca_cov took:     0.0436 seconds | [400,3200]*[3200,400] + symeig([400,400])
# pca_svd_tXU took: 0.1503 seconds | svd([400,3200]) + [3200,400]*[400,10]
# pca_svd_US took:  0.0769 seconds | svd([3200,400]) + [3200,10]*[10]
# pca_svd_XV took:  0.1200 seconds | svd([3200,400]) + [3200,400]*[400,10]

# Checking that we have the same results.
# mean_absolute_error(pca_cov, pca_svd_tXU) = 0.0024785049408249
# mean_absolute_error(pca_cov, -pca_svd_tXU) = 0.00310201526963116
# ---------------------------------------------
# mean_absolute_error(pca_cov, pca_svd_US) = 0.01826231367969895
# mean_absolute_error(pca_cov, -pca_svd_US) = 0.03964256734929569
# ---------------------------------------------
# mean_absolute_error(pca_cov, pca_svd_XV) = 0.00183384942254244
# mean_absolute_error(pca_cov, -pca_svd_XV) = 0.003746670787913525
# ---------------------------------------------

# ###########################
# Starting a new experiment

# Matrix of shape: [3200, 800]
# Target PCA:      [3200, 10]

# Hilbert matrix creation took: 0.0260 seconds
# pca_cov took:     0.0854 seconds | [800,3200]*[3200,800] + symeig([800,800])
# pca_svd_tXU took: 0.4021 seconds | svd([800,3200]) + [3200,800]*[800,10]
# pca_svd_US took:  0.2205 seconds | svd([3200,800]) + [3200,10]*[10]
# pca_svd_XV took:  0.2207 seconds | svd([3200,800]) + [3200,800]*[800,10]

# Checking that we have the same results.
# mean_absolute_error(pca_cov, pca_svd_tXU) = 0.002819157373369499
# mean_absolute_error(pca_cov, -pca_svd_tXU) = 0.003191437907843597
# ---------------------------------------------
# mean_absolute_error(pca_cov, pca_svd_US) = 2.678524170739356e+244
# mean_absolute_error(pca_cov, -pca_svd_US) = 0.07123846897028652
# ---------------------------------------------
# mean_absolute_error(pca_cov, pca_svd_XV) = 0.00189604420097131
# mean_absolute_error(pca_cov, -pca_svd_XV) = 0.004114551080241788
# ---------------------------------------------

# ###########################
# Starting a new experiment

# Matrix of shape: [3200, 1600]
# Target PCA:      [3200, 10]

# Hilbert matrix creation took: 0.0266 seconds
# pca_cov took:     0.2399 seconds | [1600,3200]*[3200,1600] + symeig([1600,1600])
# pca_svd_tXU took: 0.9850 seconds | svd([1600,3200]) + [3200,1600]*[1600,10]
# pca_svd_US took:  0.7064 seconds | svd([3200,1600]) + [3200,10]*[10]
# pca_svd_XV took:  0.7698 seconds | svd([3200,1600]) + [3200,1600]*[1600,10]

# Checking that we have the same results.
# mean_absolute_error(pca_cov, pca_svd_tXU) = 0.002985712544431292
# mean_absolute_error(pca_cov, -pca_svd_tXU) = 0.003281119841686649
# ---------------------------------------------
# mean_absolute_error(pca_cov, pca_svd_US) = 0.0852999008709448
# mean_absolute_error(pca_cov, -pca_svd_US) = 2.678524170739356e+244
# ---------------------------------------------
# mean_absolute_error(pca_cov, pca_svd_XV) = 0.001967868654544759
# mean_absolute_error(pca_cov, -pca_svd_XV) = 0.004298963731573184
# ---------------------------------------------

# ###########################
# Starting a new experiment

# Matrix of shape: [3200, 3200]
# Target PCA:      [3200, 10]

# Hilbert matrix creation took: 0.0256 seconds
# pca_cov took:     0.9037 seconds | [3200,3200]*[3200,3200] + symeig([3200,3200])
# pca_svd_tXU took: 3.5530 seconds | svd([3200,3200]) + [3200,3200]*[3200,10]
# pca_svd_US took:  3.4853 seconds | svd([3200,3200]) + [3200,10]*[10]
# pca_svd_XV took:  3.6580 seconds | svd([3200,3200]) + [3200,3200]*[3200,10]

# Checking that we have the same results.
# mean_absolute_error(pca_cov, pca_svd_tXU) = 0.0009401260244585951
# mean_absolute_error(pca_cov, -pca_svd_tXU) = 0.005445552585551723
# ---------------------------------------------
# mean_absolute_error(pca_cov, pca_svd_US) = 2.072544378468605e+262
# mean_absolute_error(pca_cov, -pca_svd_US) = nan
# ---------------------------------------------
# mean_absolute_error(pca_cov, pca_svd_XV) = 0.0009401260244585959
# mean_absolute_error(pca_cov, -pca_svd_XV) = 0.005445552585551725
# ---------------------------------------------
