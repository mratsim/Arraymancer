import ../src/arraymancer
import times, strformat

# Compile with -d:release -d:danger (optionally -d:openmp)

const
  Observations = 20000
  Features = 4000
  N = max(Observations, Features)
  k = 40

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

# i9-9980XE Overclocked at 4.1GHz, AVX 4.0GHz, AVX512 3.5GHz
# BLAS / Lapack linked with OpenBLAS
#
# $  ./benchmarks/xtime.rb ./build/randomized_svd
# Hilbert matrix creation took: 0.8678 seconds
# Matrix of shape: [20000, 4000]
# Target SVD: [20000, 40]
# Randomized SVD took: 10.7725 seconds
# U: [20000, 40]
# S: [40]
# Vh: [40, 4000]
# ---------------- xtime.rb
# 11.73s, 3094.0Mb

# i9-9980XE Overclocked at 4.1GHz, AVX 4.0GHz, AVX512 3.5GHz
# BLAS / Lapack linked with Intel MKL
#
# $  ./benchmarks/xtime.rb ./build/randomized_svd
# Hilbert matrix creation took: 0.8236 seconds
# Matrix of shape: [20000, 4000]
# Target SVD: [20000, 40]
# Randomized SVD took: 5.7280 seconds
# U: [20000, 40]
# S: [40]
# Vh: [40, 4000]
# ---------------- xtime.rb
# 6.64s, 3127.1Mb
