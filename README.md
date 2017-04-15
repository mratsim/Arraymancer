# Arraymancer

A tensor (N-dimensional array) project. Focus on machine learning, deep learning and numerical computing.

A tensor support arbitrary types (floats, strings, objects ...).

EXPERIMENTAL: API may change and break.

## Goals

The automatic backpropagation library [Nim-RMAD](https://github.com/mratsim/nim-rmad) needs to be generalized to vectors and matrices, 3D, 4D, 5D tensors to support:

* height
* width
* depth for 3D or time for video
* batch size
* RGB color channels

Unfortunately, attempts to use [linalg's](https://github.com/unicredit/linear-algebra) vector and matrix types were unsuccessful. Support for 3D+ tensors would also need more work.

This library aims to provided an efficient tensor/ndarray type. Focus will be on numerical computation (BLAS) and GPU support.
The library will be flexible enough to represent arbitrary N-dimensional Arrays, especially for NLP word vectors.

## Not prioritized

Numpy-like functionality: 
* slicing,
* iterating,
* assigning,
* statistics (mean, median, stddev ...)

will be added on a as-needed basis.