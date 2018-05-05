Arraymancer v0.4.0 May 05 2018 "The Name of the Wind"
=====================================================

Changes:

- Core:
  - OpenCL tensors are now available! However Arraymancer will naively select the first backend available. It can be CPU, it can be GPU. They support basic and broadcasted operations (Addition, matrix multiplication, elementwise multiplication, ...)
  - Addition of an `argmax` and `argmax_max` procs.

- Datasets:
  - Loading the MNIST dataset from http://yann.lecun.com/exdb/mnist/
  - Reading and writing from CSV

- Linear algebra:
  - Least squares solver
  - Eigenvalues and eigenvectors decomposition for symmetric matrices

- Machine Learning
  - Principal Component Analysis (PCA)

- Statistics
  - Computation of covariance matrices

- Neural network
  - Introduction of a short intuitive syntax to build neural networks! (A blend of Keras and PyTorch).
  - Maxpool2D layer
  - Mean Squared Error loss
  - Tanh and softmax activation functions

- Examples and tutorials
  - Digit recognition using Convolutional Neural Net
  - Teaching Fizzbuzz to a neural network

- Tooling
  - Plotting tensors through Python

Several updates linked to Nim rapid development and several bugfixes.

Thanks:
  - Bluenote10 for the CSV writing proc and the tensor plotting tool
  - Miran for benchmarking
  - Manguluka for tanh
  - Vindaar for bugfixing
  - Every participants in RFCs
  - And you user of the library.

Arraymancer v0.3.0 Dec. 14 2017 "Wizard's First Rule"
=====================================================

I am very excited to announce the third release of Arraymancer which includes numerous improvements, features and (unfortunately!) breaking changes.
Warning  ⚠: Deprecated ALL procs will be removed next release due to deprecated spam and to reduce maintenance burden.

Changes:
- **Very** Breaking
  - Tensors uses reference semantics now: `let a = b` will share data by default and copies must be made explicitly.
    - There is no need to use `unsafe` proc to avoid copies especially for slices.
    - Unsafe procs are deprecated and will be removed leading to a smaller and simpler codebase and API/documentation.
    - Tensors and CudaTensors now works the same way.
    - Use `clone` to do copies.
    - Arraymancer now works like Numpy and Julia, making it easier to port code.
    - Unfortunately it makes it harder to debug unexpected data sharing.

- Breaking (?)
  - The max number of dimensions supported has been reduced from 8 to 7 to reduce cache misses.
    Note, in deep learning the max number of dimensions needed is 6 for 3D videos: [batch, time, color/feature channels, Depth, Height, Width]

- Documentation
  - Documentation has been completely revamped and is available here: https://mratsim.github.io/Arraymancer/

- Huge performance improvements
  - Use non-initialized seq
  - shape and strides are now stored on the stack
  - optimization via inlining all higher-order functions
    - `apply_inline`, `map_inline`, `fold_inline` and `reduce_inline` templates are available.
  - all higher order functions are parallelized through OpenMP
  - integer matrix multiplication uses SIMD, loop unrolling, restrict and 64-bit alignment
  - prevent false sharing/cache contention in OpenMP reduction
  - remove temporary copies in several proc
  - runtime checks/exception are now behind `unlikely`
  - `A*B + C` and `C+=A*B` are automatically fused in one operation
  - do not initialized result tensors

- Neural network:
  - Added `linear`, `sigmoid_cross_entropy`, `softmax_cross_entropy` layers
  - Added Convolution layer

- Shapeshifting:
  - Added `unsqueeze` and `stack`

- Math:
  - Added `min`, `max`, `abs`, `reciprocal`, `negate` and in-place `mnegate` and `mreciprocal`

- Statistics:
  - Added variance and standard deviation

- Broadcasting
  - Added `.^` (broadcasted exponentiation)

- Cuda:
  - Support for convolution primitives: forward and backward
  - Broadcasting ported to Cuda

- Examples
  - Added perceptron learning `xor` function example

- Precision
  - Arraymancer uses `ln1p` (`ln(1 + x)`) and `exp1m` procs (`exp(1 - x)`) where appropriate to avoid catastrophic cancellation

- Deprecated
  - Version 0.3.1 with the ALL deprecated proc removed will be released in a week. Due to issue https://github.com/nim-lang/Nim/issues/6436,
    even using non-deprecated proc like `zeros`, `ones`, `newTensor` you will get a deprecated warning.
  - `newTensor`, `zeros`, `ones` arguments have been changed from `zeros([5, 5], int)` to `zeros[int]([5, 5])`
  - All `unsafe` proc are now default and deprecated.


Arraymancer v0.2.0 Sept. 24, 2017 "The Color of Magic"
======================================================

I am very excited to announce the second release of Arraymancer which includes numerous improvements `blablabla` ...

Without further ado:
- Communauty
   - There is a Gitter room!
- Breaking
   - `shallowCopy` is now `unsafeView` and accepts `let` arguments
   - Element-wise multiplication is now `.*` instead of `|*|`
   - vector dot product is now `dot` instead of `.*`
- Deprecated
   - All tensor initialization proc have their `Backend` parameter deprecated.
   - `fmap` is now `map`
   - `agg` and `agg_in_place` are now `fold` and nothing (too bad!)

- Initial support for Cuda !!!
   - All linear algebra operations are supported
   - Slicing (read-only) is supported
   - Transforming a slice to a new contiguous Tensor is supported
- Tensors
   - Introduction of `unsafe` operations that works without copy: `unsafeTranspose`, `unsafeReshape`, `unsafebroadcast`, `unsafeBroadcast2`, `unsafeContiguous`,
   - Implicit broadcasting via `.+, .*, ./, .-` and their in-place equivalent `.+=, .-=, .*=, ./=`
   - Several shapeshifting operations: `squeeze`, `at` and their `unsafe` version.
   - New property: `size`
   - Exporting: `export_tensor` and `toRawSeq`
   - Reduce and reduce on axis
- Ecosystem:
   - I express my deep thanks to @edubart for testing Arraymancer, contributing new functions, and improving its overall performance. He built [arraymancer-demos](https://github.com/edubart/arraymancer-demos) and [arraymancer-vision](https://github.com/edubart/arraymancer-vision),check those out you can load images in Tensor and do logistic regression on those!

Also thanks to the Nim communauty on IRC/Gitter, they are a tremendous help (yes Varriount, Yardanico, Zachary, Krux).
I probably would have struggled a lot more without the guidance of Andrea's code for Cuda in his [neo](https://github.com/unicredit/neo) and [nimcuda](https://github.com/unicredit/nimcuda) library. And obviously Araq and Dom for Nim which is an amazing language for performance, productivity, safety and metaprogramming.


Minor revisions v0.1.1 to v0.1.3
================================

Arraymancer v0.1.0 July 12, 2017 "Magician Apprentice"
=======================================================

First public release.

Include:

- converting from deep nested proc or array
- Slicing, and slice mutation
- basic linear algebra operations,
- reshaping, broadcasting, concatenating,
- universal functions
- iterators (in-place, axis, inline and closure versions)
- BLAS and BLIS support for fast linear algebra
