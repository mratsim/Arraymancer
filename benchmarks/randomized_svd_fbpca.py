import numpy as np
import time
from fbpca import pca
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
    (U, S, Vh) = pca(H, k=k, raw=True, n_iter=2, l=k+5) # Raw=True for SVD
    stop = time.time()

    print(f'Randomized SVD took: {stop-start:>4.4f} seconds')
    print("U: ", U.shape)
    print("S: ", S.shape)
    print("Vh: ", Vh.shape)

    print("---------------------------------------------------------------------------------")

bench(Observations = 20000, Features = 4000)
bench(Observations = 4000, Features = 20000)

# i9-9980XE Overclocked at 4.1GHz, AVX 4.0GHz, AVX512 3.5GHz
# Numpy / Scipy built with OpenBLAS
#
# Hilbert matrix creation too: 0.9500 seconds.
# Matrix of shape: [20000, 4000]
# Target SVD: [20000, 40]
# Randomized SVD took: 2.1240 seconds
# U:  (20000, 40)
# S:  (40,)
# Vh:  (40, 4000)
# ---------------------------------------------------------------------------------
# Hilbert matrix creation too: 0.9441 seconds.
# Matrix of shape: [4000, 20000]
# Target SVD: [4000, 40]
# Randomized SVD took: 0.4008 seconds
# U:  (4000, 40)
# S:  (40,)
# Vh:  (40, 20000)
# ---------------------------------------------------------------------------------
# 4.81s, 3742.4Mb  -- xtime.rb

# mem usage with just the first SVD: 3.77GB

# Numpy / Scipy built with MKL
#
# Hilbert matrix creation too: 0.9380 seconds.
# Matrix of shape: [20000, 4000]
# Target SVD: [20000, 40]
# Randomized SVD took: 2.1632 seconds
# U:  (20000, 40)
# S:  (40,)
# Vh:  (40, 4000)
# ---------------------------------------------------------------------------------
# Hilbert matrix creation too: 0.9698 seconds.
# Matrix of shape: [4000, 20000]
# Target SVD: [4000, 40]
# Randomized SVD took: 0.3566 seconds
# U:  (4000, 40)
# S:  (40,)
# Vh:  (40, 20000)
# ---------------------------------------------------------------------------------
# 4.78s, 3742.7Mb
