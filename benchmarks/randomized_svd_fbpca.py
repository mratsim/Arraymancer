import numpy as np
import time
from fbpca import pca
from scipy.linalg import hilbert

Observations = 20000
Features = 4000
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

# i9-9980XE Overclocked at 4.1GHz, AVX 4.0GHz, AVX512 3.5GHz
# Numpy / Scipy built with MKL
#
# Hilbert matrix creation too: 0.9457 seconds.
# Matrix of shape: [20000, 4000]
# Target SVD: [20000, 40]
# Randomized SVD took: 2.1181 seconds
# U:  (20000, 40)
# S:  (40,)
# Vh:  (40, 4000)
# ---------------- xtime.rb
# 3.35s, 3703.2Mb
