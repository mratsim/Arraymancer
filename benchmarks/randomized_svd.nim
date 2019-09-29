import ../src/arraymancer
import times, strformat

# Compile with -d:release -d:danger (optionally -d:openmp)

proc bench(Observations, Features: static int) =
  const N = max(Observations, Features)
  const k = 40

  var start, stop: float64
  # Create a known ill-conditionned matrix for testing
  # This requires 20k * 20k * 4 bytes (float32) = 1.6 GB

  start = epochTime()
  let H = hilbert(N, float32)[0..<Observations, 0..<Features]
  stop = epochTime()

  echo &"Hilbert matrix creation took: {stop-start:>4.4f} seconds"
  echo &"Matrix of shape: [{Observations}, {Features}]"
  echo &"Target SVD: [{Observations}, {k}]"

  block: # Random SVD
    start = epochTime()
    let (U, S, Vh) = svd_randomized(H, n_components=k, n_oversamples=5, n_power_iters=2)
    stop = epochTime()
    echo &"Randomized SVD took: {stop-start:>4.4f} seconds"
    echo "U: ", U.shape
    echo "S: ", S.shape
    echo "Vh: ", Vh.shape
  echo "---------------------------------------------------------------------------------"

bench(Observations = 20000, Features = 4000)
bench(Observations = 4000, Features = 20000)

# i9-9980XE Overclocked at 4.1GHz, AVX 4.0GHz, AVX512 3.5GHz
# BLAS / Lapack linked with OpenBLAS
#
# $  ./benchmarks/xtime.rb ./build/rsvd
# Hilbert matrix creation took: 0.8458 seconds
# Matrix of shape: [20000, 4000]
# Target SVD: [20000, 40]
# Randomized SVD took: 2.7272 seconds
# U: [20000, 40]
# S: [40]
# Vh: [40, 4000]
# ---------------------------------------------------------------------------------
# Hilbert matrix creation took: 0.8386 seconds
# Matrix of shape: [4000, 20000]
# Target SVD: [4000, 40]
# Randomized SVD took: 0.1279 seconds
# U: [4000, 40]
# S: [40]
# Vh: [40, 45]
# ---------------------------------------------------------------------------------
# 4.68s, 4567.4Mb  -- xtime.rb

# #################################################################################
# i9-9980XE Overclocked at 4.1GHz, AVX 4.0GHz, AVX512 3.5GHz
# BLAS / Lapack linked with Intel MKL
#
# $  ./benchmarks/xtime.rb build/rsvd
# Hilbert matrix creation took: 0.8093 seconds
# Matrix of shape: [20000, 4000]
# Target SVD: [20000, 40]
# Randomized SVD took: 0.3683 seconds
# U: [20000, 40]
# S: [40]
# Vh: [40, 4000]
# ---------------------------------------------------------------------------------
# Hilbert matrix creation took: 0.8181 seconds
# Matrix of shape: [4000, 20000]
# Target SVD: [4000, 40]
# Randomized SVD took: 0.1190 seconds
# U: [4000, 40]
# S: [40]
# Vh: [40, 45]
# ---------------------------------------------------------------------------------
# 2.26s, 4593.1Mb  -- xtime.rb

# Mem usage with just the first SVD: 3.1GB
