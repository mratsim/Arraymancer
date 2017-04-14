# Arraymancer

A tensor (N-dimensional array) project for machine learning and deep learning.

## Goals

The automatic backpropagation library nim-rmad needs to be generalized to vector and matrices.

Unfortunately, attempts to use [linalg's](https://github.com/unicredit/linear-algebra) vector and matrix types were unsuccessful.

This library aims to provided a tensor type for Deep Learning. Focus will be on BLAS and GPU support, 1D-Tensor (vector), 2D-Tensor (matrices), 3D and 4D Tensor (for 2d images + color channel + batch image processing).
The library will be flexible enough to represent Nd-Array, especially for NLP word vectors.

## Not prioritized

Numpy is the reference for ndarrays, numpy-like functionality (slicing, iterating) will be added on a as-needed basis.