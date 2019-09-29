import numpy as np
import time
from sklearn.utils.extmath import randomized_svd
from scipy.linalg import hilbert

def bench(Observations, Features):
    N = max(Observations, Features)
    k = 40

    # Create a known ill-conditionned matrix for testing
    # This requires 20k * 20k * 4 bytes (float32) = 1.6 GB
    start = time.time()
    H = hilbert(N)[:Observations, :Features]
    stop = time.time()

    print(f'Hilbert matrix creation too: {stop-start:>4.4f} seconds.')
    print(f'Matrix of shape: [{Observations}, {Features}]')
    print(f'Target SVD: [{Observations}, {k}]')

    start = time.time()
    (U, S, Vh) = randomized_svd(H, n_components=k, n_oversamples=5, n_iter=2)
    stop = time.time()

    print(f'Randomized SVD took: {stop-start:>4.4f} seconds')
    print("U: ", U.shape)
    print("S: ", S.shape)
    print("Vh: ", Vh.shape)

    print("---------------------------------------------------------------------------------")

bench(Observations = 20000, Features = 4000)
# bench(Observations = 4000, Features = 20000)

# i9-9980XE Overclocked at 4.1GHz, AVX 4.0GHz, AVX512 3.5GHz
# Numpy / Scipy built with MKL
#
# $  ./benchmarks/xtime.rb python benchmarks/randomized_svd_sklearn.py
# Hilbert matrix creation too: 0.9479 seconds.
# Matrix of shape: [20000, 4000]
# Target SVD: [20000, 40]
# Randomized SVD took: 3.6055 seconds
# U:  (20000, 40)
# S:  (40,)
# Vh:  (40, 4000)
# ---------------------------------------------------------------------------------
# Hilbert matrix creation too: 0.9452 seconds.
# Matrix of shape: [4000, 20000]
# Target SVD: [4000, 40]
# Randomized SVD took: 0.2505 seconds
# U:  (4000, 40)
# S:  (40,)
# Vh:  (40, 20000)
# ---------------------------------------------------------------------------------
# 6.14s, 3733.8Mb  -- xtime.rb

# mem usage with just the first SVD: 3.77GB
