Arraymancer v0.7.x
=====================================================

Arraymancer v0.7.21 Aug. 31 2023
=====================================================

- use `system.newSeqUninit` if available (PR #589)
- add support to load Fashion MNIST dataset (PR #590)

Arraymancer v0.7.20 Jun. 12 2023
=====================================================

- update `std/math` path (PR #588)
- update VM images (PR #588)

Arraymancer v0.7.19 Dec. 10 2022
=====================================================

- change the signature of `numerical_gradient` for scalars to
  explicitly reject `Tensor` arguments. See discussion in PR #580.

Arraymancer v0.7.18 Dec. 10 2022
=====================================================

- import `math` in one test to avoid regression due to upstream
  change, PR #580

Arraymancer v0.7.17 Dec. 02 2022
=====================================================

- change `KnownSupportsCopyMem` to a `concept` to allow usage of
  `supportsCopyMem` instead of a fixed list of supported types
- remove the workaround in `toTensor` as parts of it triggered an
  upstream ARC bug in certain code. The solution gets rid of the
  existing workaround and replaces it by a saner solution.

Arraymancer v0.7.16 Nov. 29 2022
=====================================================

- replace usages of `varargs` by `openArray` thanks to upstream fixes
  (PR #572)
- remove usages of dataArray (PRs #569 and #570)
- fix issue displaying some 2D tensors (PR #567)
-
- add `clone` operation for k-d tree (PR #565)
- remove `TensorHelper` type in k-d tree implementation, instead
  `bind` a custom `<` (PR #565)
- fix `deepCopy` for tensors (PR #565)
- misc fixes for current devel with ORC (PR #573)
- replace badly defined `contains` check of integer in `set[uint16]`
  (PR #575)
- unify all usages of identifiers with different capitalizations to
  remove many compile time hints / warnings and add
  `styleCheck:usages` to `nim.cfg` (PR #564)

Arraymancer v0.7.15 Jul. 28 2022
=====================================================

- replace explicit `optimizer*` procedures by generic `optimizer` with
a typed first argument `optimizer(SGD, ...)`, `optimizer(Adam, ...)`
etc. (PR #557)

Arraymancer v0.7.14 Jul. 25 2022
=====================================================

- replace `shallowCopy` by `move` under ARC/ORC (PR #562)

Arraymancer v0.7.13 Jul. 10 2022
=====================================================

- rewrote neural network DSL to support custom non-macro layers and composition of multiple models (PR #548)
- syntactic changes to the neural network DSL (PR #548)
  - autograd context is no longer specified at network definition. It is only used when instantiating a model. E.g.: `network FizzBuzzNet: ... ` instead of `network ctx, FizzBuzzNet: ...`
  - there is no `Input` layer anymore. All input variables are specified at the beginning of the `forward` declaration
  - in/output-shapes of layers are generally described as `seq[int]` now, but this depends on the concrete layer type
  - layer descriptions behave like functions, so function parameters can be specifically named and potentially reordered or omitted. E.g.: `Conv2D(@[1, 28, 28], out_channels = 20, kernel_size = (5, 5))`
  - the `Conv2D` layer expects the kernel size as a `Size2D` (integer tuple of size 2), instead of passing height and width as separate arguments
  - when using an `out_shape` function of a previous layer to describe a `Linear` layer, one has to use `out_shape[0]` for the number of input features. E.g.: `Linear(fl.out_shape[0], 500)` instead of `Linear(fl.out_shape, 500)`
  - `GRU` has been replaced by `GRULayer`, which has now a different description signature: `(num_input_features, hidden_size, stacked_layers)` instead of `([seq_len, Batch, num_input_features], hidden_size, stacked_layers)`

Arraymancer v0.7.12 Apr. 12 2022
=====================================================

- Remove cursor annotation for CPU Tensor again as it causes memory
  leaks (PR #555)
- disable OpenMP in reducing contexts if ARC/ORC is used as it leads
  to segfaults (PR #555)

Arraymancer v0.7.11 Feb. 22 2022
=====================================================

- Add cursor annotation for CPU Tensor fixing #535 (PR #533)
- Add `toFlatSeq` (flatten data and export it as a seq) and `toSeq1D`
  which export a rank-1 Tensor into a `seq` (PR #533)
- add test for `syevr` (PR #552)
- introduce CI for Nim 1.6 and switch to Github Action task for Nim
  binaries (PR #551)

Arraymancer v0.7.10 Dec. 30 2021
=====================================================

- fixes `einsum` in more general generic contexts by replacing the
  logic introduced in PR #539 by a rewrite from typed AST to
  `untyped`. This makes it work in (all?) generic contexts, PR #545
- add element wise exponentiation `^.` for scalar base to all elements
  of a tensor, i.e. `2^.t` for a tensor `t`, thanks to @asnt PR #546.

Arraymancer v0.7.9 Dec. 29 2021
=====================================================

- fixes `einsum` in generic contexts by allowing `nnkCall` and
  `nnkOpenSymChoice`, PR #539
- add test for `einsum` showing cross product, PR #538
- add test for display of uninitialized tensor, PR #540
- allow `CustomMetric` as user defined metric in `distances`, PR #541. User
  must provide their own `distance` procedure for the metric with
  signature:
  ```nim
  proc distance*(metric: typedesc[CustomMetric], v, w: Tensor[float]): float
  ```
- disable AVX512 support by default. Add the `-d:avx512` compilation
  flag to activate it. Note: this activates it for all CPUs as it
  hands `-mavx512dq` to gcc / clang!

Arraymancer v0.7.8 Oct. 27 2021
=====================================================

- further fix undeclared identifier issues present for certain
  generics context, in this case for the `|` identifier when slicing
- fix printing of uninitialized tensors. Instead of crashing these now
  print as "Unitialized Tensor[T] ...".
- fix CSV parsing regression (still used `Fdata` field access) and
  improved efficiency of the parser

Arraymancer v0.7.7 Oct. 14 2021
=====================================================

- fix autograd code after changes in the Nim compiler from version 1.4
  on. Requires to keep the procedure signatures using the base type
  and then convert to the specific type in the procedure (#528)
- fixes an issue when creating a laser tensor within a generic, in
  which case one might see "undeclared identifier `rank`"


Arraymancer v0.7.6 Aug. 13 2021
=====================================================

- remove the MNIST download test from the default test cases, as it is
  too flaky for a CI
- fix issue #523 by fixing how `map_inline` works for rank > 1
  tensors. Access the data storage at correct indices instead of using
  the regular tensor `[]` accessor (which requires N indices for a
  rank N tensor)


Arraymancer v0.7.5 Jul. 26 2021
=====================================================

- change least squares wrapper around `gelsd` to have workspace
  computed by LAPACK and set `rcond` to default `-1`. But we also make
  it an argument for users to change (PR #520)


Arraymancer v0.7.4 Jul. 20 2021
=====================================================

- add k-d tree (PR #447)
- add DBSCAN clustering (PR #413 / PR #518), thanks to @abieler
- `argsort` now has a `toCopy` argument to avoid sorting the argument
  tensor in place if not desired
- `cumsum`, `cumprod` now use axis 0 by default if no axis given
- as part of DBSCAN + k-d tree a `distances.nim` submodule was added
  that adds Manhattan, Minkowski, Euclidean and Jaccard distances
- `nonzero` was added to get indices of non zero elements in a tensor

Arraymancer v0.7.3 Jul. 11 2021
=====================================================

- fix memory allocation to not zero initialize the memory for tensors
  (which we do manually). This made `newTensorUninit` not do what it
  was supposed to (PR #517).
- add `vandermonde` matrix constructor (PR #519)
- change `rcond` argument to `gelsd` for linear least squares solver
  to use simple `epsilon` (PR #519)

Arraymancer v0.7.2 Jul. 5 2021
=====================================================

- fixes issue #459, ambiguity in `tanh` activation layer
- fixes issue #514, make all examples compile again
- compile all examples during CI to avoid further regressions at a
  compilation level (thanks @Anuken)


Arraymancer v0.7.1 Jul. 4 2021
=====================================================

Hotfix release fixing CUDA tensor printing. The code as pushed in #509
was broken.

Arraymancer v0.7.0 Jul. 4 2021 "Memories of Ice"
=====================================================

> This release is named after "Memories of Ice" (2001), the third book of Steven Erikson epic dark fantasy masterpiece "The Malazan Book of the Fallen".

Changes :
  - Add ``toUnsafeView`` as replacement of ``dataArray`` to return a ``ptr UncheckedArray``
  - Doc generation fixes
  - `cumsum`, `cumprod`
  - Fix least square solver
  - Fix mean square error backpropagation
  - Adapt to upstream symbol resolution changes
  - Basic Graph Convolution Network
  - Slicing tutorial revamp
  - n-dimensional tensor pretty printing
  - Compilation fixes to handle Nim v1.0 to Nim devel

Thanks to @Vindaar for maintaining the repo, the docs, pretty-printing and answering many many questions on Discord while I took a step back.
Thanks to @filipeclduarte for the cumsum/cumprod, @Clonkk for updating raw data accesses, @struggle for finding a bug in mean square error backprop,
@timotheecour for spreading new upstream requirements downstream and @anon767 for Graph Neural Network.

Arraymancer v0.6.2 Dec. 22 2020
=====================================================

Changes :
  - Fancy Indexing (#434)
  - ``argmax_max`` obsolete assert to 1D/2D removed. It refered to issues #183 and merged PR #171

Deprecation
  - The dot in broadcasting and elementwise operators has changed place. This was not propagated to logical comparison
    Use `==.`, `!=.`, `<=.`, `<.`, `>=.`, `>.`
    instead of the previous order `.==`.
    This allows the broadcasting operators to have the same precedence as the
    natural operators.
    This also align Arraymancer with other Nim packages: Manu and NumericalNim

Overview (TODO)

This release integrates part of the Laser Backend (https://github.com/numforge/laser) that has been brewing since the end of 2018. The new backend provides the following features:
- Tensors can now either be a view over a memory buffer or manage the memory (like before).
  The "view" allows zero-copy with libraries using the same
  multi-dimensional array/tensor memory layout in particular Numpy,
  PyTorch and Tensorflow or even image libraries. This can be achieved
  using the new `fromBuffer` procedures to create a tensor.
- strings, ref types and types with non-trivial destructors will still always own and manager their memory buffer.
  Trivial types (plain-old data) like integers, floats or complex can use the zero-copy scheme
  by setting `isMemOwner` to false and then point `raw_buffer` to the preallocated buffer.
  In that case, the memory must be freed manually to avoid memory leaks.
- To keep the benefits of enforcing (im)mutabilility via the type system, procedures like `dataArray` that used to return raw pointers have been deleted or deprecated in favor of
  routines that return `RawImmutableView` and `RawMutableView` with only appropriate
  indexing or mutable indexing defined.
  This is an improvement over raw pointers.
  Note that at the moment there is no scheme like a borrow-checker to prevent users from using them even after the buffer has been invalidated (borrow-checking).
  In the future `lent` will be used to provide borrow-checking security.

Breaking changes
- In the past, it was mentioned in the README that Arraymancer supported up to 6 dimensions.
  In reality up to 7 dimensions was possible. It has now been changed to 6 by default.
  It is now possible to configure this via a compiler define `LASER_MAXRANK`
  For example `nim c -d:LASER_MAXRANK=16 path/to/app` to support up to 16 dimensions.
  or `nim c -d:LASER_MAXRANK=2 path/to/app` if only 2 dimensions are ever needed and we want to save on stack space and optimize memory cache accesses.
- The CpuStorage data structure has been completely refactored
- The routines `data`, `data=` and `toRawSeq` that used to return the `seq` backing the Tensor
  have been changed in a backward-incompatible way. They now return the canonical row-major
  representation of a tensor. With the change to a view and decoupling with a lower-level pointer based backend, Arraymancer does not track anymore the whole reserved memory
  and so cannot return the raw in-memory storage of the tensor.
  They have been deprecated.
- Some procedures now have side-effects inherited from Nim's `allocShared`
  - `variable` from the `autograd` module
  - `solve` from the `linear_algebra` module
- `io_hdf5` is not imported automatically anymore if the module is
  installed. The reason for this is that the HDF5 library runs code in
  global scope to initialize the HDF5 library. This means dead code
  elimination does not work and a binary will always depend on the
  HDF5 shared library if the `nimhdf5` is installed, even if not used.
  Simply import using `import arraymancer/io/io_hdf5`.

Deprecation
- `MetadataArray` is now `Metadata`
- `dataArray` has been deprecated in favor on mutability-safe
  `unsafe_raw_offset`

Arraymancer v0.6.0 Jan. 09 2020 - "Windwalkers"
=====================================================

> This release is named after "Windwalkers" (2004, French "La Horde du Contrevent"), by Alain Damasio.

Changes:
  - The ``symeig`` proc to compute eigenvectors of a symmetric matrix
    now accepts an "uplo" char parameter. This allows to fill only the Upper or Lower
    part of the matrix, the other half is not used in computation.
  - Added ``svd_randomized``, a fast and accurate SVD approximation via random sampling.
    This is the standard driver for large scale SVD applications as SVD on large matrices is very slow.
  - ``pca`` now uses the randomized SVD instead of computing the covariance matrix.
    It can now efficiently deal with large scale problems.
    It now accepts a ``center``, ``n_oversamples`` and ``n_power_iters`` arguments.
    Note that ``pca`` without centering is equivalent to a truncated SVD.
  - LU decomposition has been added
  - QR decomposition has been added
  - ``hilbert`` has been introduced. It creates the famous ill-conditioned Hilbert matrix.
    The matrix is suitable to stress test decompositions.
  - The ``arange`` procedure has been introduced. It creates evenly spaced value within a specified range
    and step
  - The ordering of arguments to error functions has been converted to
    `(y_pred, y_target)` (from (y_target, y_pred)), enabling the syntax `y_pred.accuracy_score(y)`.
    All existing error functions in Arraymancer were commutative w.r.t. to arguments
    so existing code will keep working.
  - a ``solve`` procedure has been added to solve linear system of equations represented as matrices.
  - a ``softmax`` layer has been added to the autograd and neural networks
    complementing the SoftmaxCrossEntropy layer which fused softmax + Negative-loglikelihood.
  - The stochastic gradient descent now has a version with Momentum

Bug fixes:
  - ``gemm`` could crash when the result was column major.
  - The automatic fusion of matrix multiplication with matrix addition `(A * X) + b` could update the b matrix.
  - Complex converters do not pollute the global namespace and do not
    prevent string covnersion via `$` of number types due to ambiguous call.
  - in-place division has been fixed, a typo made it into substraction.
  - A conflict between NVIDIA "nanosecond" and Nim times module "nanosecond"
    preventing CUDA compilation has been fixed

Breaking
  - In ``symeig``, the ``eigenvectors`` argument is now called ``return_eigenvectors``.
  - In ``symeig`` with slice, the new ``uplo`` precedes the slice argument.
  - pca input "nb_components" has been renamed "n_components".
  - pca output tuple used the names (results, components). It has been renamed to (projected, components).
  - A ``pca`` overload that projected a data matrix on already existing principal axes
    was removed. Simply multiply the mean-centered data matrix with the loadings instead.
  - Complex converters were removed. This prevents hard to debug and workaround implicit conversion bug in downstream library.
    If necessary, users can reimplement converters themselves.
    This also provides a 20% boost in Arraymancer compilation times

Deprecation:
  - The syntax gemm(A, B, C) is now deprecated.
    Use explicit "gemm(1.0, A, B, 0.0, C)" instead.
    Arguably not zero-ing C could also be a reasonable default.
  - The dot in broadcasting and elementwise operators has changed place
    Use `+.`, `*.`, `/.`, `-.`, `^.`, `+.=`, `*.=`, `/.=`, `-.=`, `^.=`
    instead of the previous order `.+` and `.+=`.
    This allows the broadcasting operators to have the same precedence as the
    natural operators.
    This also align Arraymancer with other Nim packages: Manu and NumericalNim

Thanks to @dynalagreen for the SGD with Momentum,  @xcokazaki for spotting the in-place division typo,
@Vindaar for fixing the automatic matrix multiplication and addition fusion,
@Imperator26 for the Softmax layer, @brentp for reviewing and augmenting the SVD and PCA API,
@auxym for the linear equation solver and @berquist for the reordering all error functions to the new API.
Thanks @b3liever for suggesting the dot change to solve the precedence issue in broadcasting and elementwise operators.

Arraymancer v0.5.1 Jul. 19 2019
=====================================================
Changes affecting backward compatibility:
  - None

Changes:
  - 0.20.x compatibility (commit 0921190)
  - Complex support
  - `Einsum`
  - Naive whitespace tokenizer for NLP
  - Preview of Laser backend for matrix multiplication without SIMD autodetection (already 5x faster on integer matrix multiplication)

Fix:
  - Fix height/width order when reading an image in tensor

Thanks to @chimez for the complex support and updating for 0.20, @metasyn for the tokenizer,
@xcokazaki for the image dimension fix and @Vindaar for the einsum implemention

Arraymancer v0.5.0 Dec. 23 2018 "Sign of the Unicorn"
=====================================================

> This release is named after "Sign of the Unicorn" (1975), the third book of Roger Zelazny masterpiece "The Chronicles of Amber".

Changes affecting backward compatibility:
  - PCA has been split into 2
    - The old PCA with input `pca(x: Tensor, nb_components: int)` now returns a tuple
      of result and principal components tensors in descending order instead of just a result
    - A new PCA `pca(x: Tensor, principal_axes: Tensor)` will project the input x
      on the principal axe supplied

Changes:
  - Datasets:
    - MNIST is now autodownloaded and cached
    - Added IMDB Movie Reviews dataset
  - IO:
    - Numpy file format support
    - Image reading and writing support (jpg, bmp, png, tga)
    - HDF5 reading and writing
  - Machine learning
    - Kmeans clustering
  - Neural network and autograd:
    - Support substraction, sum and stacking in neural networks
    - Recurrent NN: GRUCell, GRU and Fused Stacked GRU support
    - The NN declarative lang now supports GRU
    - Added Embedding layer with up to 3D input tensors [batch_size, sequence_length, features] or [sequence_length, batch_size, features]. Indexing can be done with any sized integers, byte or chars and enums.
    - Sparse softmax cross-entropy now supports target tensors with indices of type: any size integers, byte, chars or enums.
    - Added ADAM optimiser (Adaptative Moment Estimation)
    - Added Hadamard product backpropagation (Elementwise matrix multiply)
    - Added Xavier Glorot, Kaiming He and Yann Lecun weight initialisations
    - The NN declarative lang automatically initialises weights with the following scheme:
      - Linear and Convolution: Kaiming (suitable for Relu activation)
      - GRU: Xavier (suitable for the internal tanh and sigmoid)
      - Embedding: Not supported in declarative lang at the moment
  - Tensors:
    - Add tensor splitting and chunking
    - Fancy indexing via `index_select`
    - division broadcasting, scalar division and multiplication broadcasting
    - High-dimensional `toSeq` exports
  - End-to-end Examples:
    - Sequence/mini time-series classification example using RNN
    - Training and text generation example with Shakespeare and Jane Austen work. This can be applied to any text-based dataset (including blog posts, Latex papers and code). It should contain at least 700k characters (0.7 MB), this is considered small already.

- Important fixes:
  - Convolution shape inference on non-unit strided convolutions
  - Support the future OpenMP changes from nim#devel
  - GRU: inference was squeezing all singleton dimensions instead of just the "layer" dimension.
  - Autograd: remove pointers to avoid pointing to wrong memory when the garbage collector moves it under pressure. This unfortunately comes at the cost of more GC pressure, this will be addressed in the future.
  - Autograd: remove all methods. They caused issues with generic instantiation and object variants.

Special thanks to [@metasyn](https://github.com/metasyn) (MNIST caching, IMDB dataset, Kmeans) and [@Vindaar](https://github.com/vindaar) (HDF5 support and the example of using Arraymancer + Plot.ly) for their large contributions on this release.

Ecosystem:
  - Using Arraymancer + Plotly for NN training visualisation:
    https://github.com/Vindaar/NeuralNetworkLiveDemo
    ![](https://github.com/Vindaar/NeuralNetworkLiveDemo/raw/master/media/demo.gif)
  - [Monocle](https://github.com/numforge/monocle), proof-of-concept data visualisation in Nim using [Vega](http://vega.github.io/). Hopefully allowing this kind of visualisation in the future:

    ![](https://vega.github.io/images/vega-lite.png)
    ![](https://vega.github.io/images/vg.png)

    and compatibility with the Vega ecosystem, especially the Tableau-like [Voyager](https://github.com/vega/voyager).
  - [Agent Smith](https://github.com/numforge/agent-smith), reinforcement learning framework.
    Currently it wraps the `Arcade Learning Environment` for practicing reinforcement learning on Atari games.
    In the future it will wrap Starcraft 2 AI bindings
    and provides a high-level interface and examples to reinforcement learning algorithms.
  - [Laser](https://github.com/numforge/laser), the future Arraymancer backend
    which provides:
      - SIMD intrinsics
      - OpenMP templates with fine-grained control
      - Runtime CPU features detection for ARM and x86
      - A proof-of-concept JIT Assembler
      - A raw minimal tensor type which can work as a view to arbitrary buffers
      - Loop fusion macros for iteration on an arbitrary number of tensors.
        As far as I know it should provide the fastest multi-threaded
        iteration scheme on strided tensors all languages and libraries included.
      - Optimized reductions, exponential and logarithm functions reaching
        4x to 10x the speed of naively compiled for loops
      - Optimised parallel strided matrix multiplication reaching 98% of OpenBLAS performance
        - This is a generic implementation that can also be used for integers
        - It will support preprocessing (relu_backward, tanh_backward, sigmoid_backward)
          and epilogue (relu, tanh, sigmoid, bias addition) operation fusion
          to avoid looping an extra time with a memory bandwidth bound pass.
      - Convolutions will be optimised with a preprocessing pass fused into matrix multiplication. Traditional `im2col` solutions can only reach 16% of matrix multiplication efficiency on the common deep learning filter sizes
      - State-of-the art random distributions and random sampling implementations
        for stochastic algorithms, text generation and reinforcement learning.

Future breaking changes.

1.  Arraymancer backend will switch to `Laser` for next version.
    Impact:
      - At a low-level CPU tensors will become a view on top of a pointer+len
        fon old data types instead of using the default Nim seqs. This will enable plenty of no-copy use cases
        and even using memory-mapped tensors for out-of-core processing.
        However libraries relying on teh very low-level representation of tensors will break.
        The future [type is already implemented in Laser](https://github.com/numforge/laser/blob/553497e1193725522ab7a5540ed824509424992f/laser/tensor/datatypes.nim#L12-L30).
      - Tensors of GC-allocated types like seq, string and references will keep using Nim seqs.
      - While it was possible to use the Javascript backend by modifying the iteration scheme
        this will not be possible at all. Use JS->C FFI or WebAssembly compilation instead.
      - The inline iteration **templates** `map_inline`, `map2_inline`, `map3_inline`, `apply_inline`, `apply2_inline`, `apply3_inline`, `reduce_inline`, `fold_inline`, `fold_axis_inline` will be removed and replace by `forEach` and `forEachStaged` with the following syntax:
      ```Nim
      forEach x in a, y in b, z in c:
        x += y * z
      ```
      Both will work with an arbitrary number of tensors and will generate 2x to 3x more compact code wile being about 30% more efficient for strided iteration. Furthermore `forEachStaged` will allow precise control of the parallelisation strategy including pre-loop and post-loop synchronisation with thread-local variables, locks, atomics and barriers.
      The existing higer-order **functions** `map`, `map2`, `apply`, `apply2`, `fold`, `reduce` will not be impacted. For small inlinable functions it will be recommended to use the `forEach` macro to remove function call overhead (Yyou can't inline a proc parameter).

2. The neural network domain specific language will use less magic
    for the `forward` proc.
    Currently the neural net domain specific language only allows the type
    `Variable[T]` for inputs and the result.
    This prevents its use with embedding layers which also requires an index input.
    Furthermore this prevents using `tuple[output, hidden: Variable]` result type
    which is very useful to pass RNNs hidden state for generative neural networks (for example text sequence or time-series).
    So unfortunately the syntax will go from the current
    `forward x, y:` shortcut to classic Nim `proc forward[T](x, y: Variable[T]): Variable[T]`

3. Once CuDNN GRU is implemented, the GRU layer might need some adjustments to give the same results on CPU and Nvidia's GPU and allow using GPU trained weights on CPU and vice-versa.

Thanks:
  - metasyn: Datasets and Kmeans clustering
  - vindaar: HDF5 support and Plot.ly demo
  - bluenote10: toSeq exports
  - andreaferetti: Adding axis parameter to Mean layer autograd
  - all the contributors of fixes in code and documentation
  - the Nim community for the encouragements

Arraymancer v0.4.0 May 05 2018 "The Name of the Wind"
=====================================================

> This release is named after "The Name of the Wind" (2007), the first book of Patrick Rothfuss masterpiece "The Kingkiller Chronicle".

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

> This release is named after "Wizard's First Rule" (1994), the first book of Terry Goodkind masterpiece "The Sword of Truth".

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


Arraymancer v0.2.0 Sept. 24, 2017 "The Colour of Magic"
======================================================

> This release is named after "The Colour of Magic" (1983), the first book of Terry Pratchett masterpiece "Discworld".

I am very excited to announce the second release of Arraymancer which includes numerous improvements `blablabla` ...

Without further ado:
- community
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

Also thanks to the Nim community on IRC/Gitter, they are a tremendous help (yes Varriount, Yardanico, Zachary, Krux).
I probably would have struggled a lot more without the guidance of Andrea's code for Cuda in his [neo](https://github.com/unicredit/neo) and [nimcuda](https://github.com/unicredit/nimcuda) library. And obviously Araq and Dom for Nim which is an amazing language for performance, productivity, safety and metaprogramming.


Minor revisions v0.1.1 to v0.1.3
================================

Arraymancer v0.1.0 July 12, 2017 "Magician Apprentice"
=======================================================

> This release is named after "Magician: Apprentice" (1982), the first book of Raymond E. Feist masterpiece "The Riftwar Cycle".

First public release.

Include:

- converting from deep nested proc or array
- Slicing, and slice mutation
- basic linear algebra operations,
- reshaping, broadcasting, concatenating,
- universal functions
- iterators (in-place, axis, inline and closure versions)
- BLAS and BLIS support for fast linear algebra
