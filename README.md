# Arraymancer

A tensor (N-dimensional array) project. Focus on machine learning, deep learning and numerical computing.

A tensor supports arbitrary types (floats, strings, objects ...).

EXPERIMENTAL: API may change and break.

## Goals

The automatic backpropagation library [Nim-RMAD](https://github.com/mratsim/nim-rmad) needs to be generalized to vectors and matrices, 3D, 4D, 5D tensors for deep learning.

Unfortunately, attempts to use [linalg's](https://github.com/unicredit/linear-algebra) vector and matrix types were unsuccessful. Support for 3D+ tensors would also need more work.

This library aims to provided an efficient tensor/ndarray type. Focus will be on numerical computation (BLAS) and GPU support.
The library will be flexible enough to represent arbitrary N-dimensional Arrays, especially for NLP word vectors.

## Current status

EXPERIMENTAL: Arraymancer may summon Ragnarok and cause the heat death of the Universe.

Arraymancer's tensors currently support the following:
* Wrapping any type: string, floats, object
* Getting and setting value at a specific index (Caveat: negative indices support needs work)
* Creating a tensor from deep nested sequences
* Universal functions from Nim math module: cos, ln, sqrt... will work element-wise
* Creating your own universal functions with `makeUniversal`, `makeUniversalLocal` and `fmap`.
    
    `fmap` can even be used on functions with input/ouput of different types.
* Optimized Linear Algebra through BLAS (via [nimblas](https://github.com/unicredit/nimblas))
    
    For now only Matrix to Matrix multiplication is available, multiplication and addition for Vector-Vector, Matrix-Vector and Matrix-Matrix are coming very soon.

Check syntax examples in the test folder.

## Not prioritized

The following Numpy-like functionality: 
* slicing,
* iterating,
* assigning,
* statistics (mean, median, stddev ...)

will be added on an as-needed basis.