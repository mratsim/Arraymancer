Arraymancer v0.x.x
=====================================================

Changes:
  - The ``symeig`` proc to compute eigenvectors of a symmetric matrix
    now accepts an "uplo" char parameter. This allows to fill only the Upper or Lower
    part of the matrix, the other half is not used in computation.
  - Added ``svd_randomized``, a fast and accurate SVD approximation via random sampling.
    This is the standard driver for large scale SVD applications as SVD on larg matrix is very slow.
  - ``pca`` now uses the randomized SVD instead of computing the covariance matrix.
    It can now efficiently deal with large scale problems.
    It now accepts a ``center``, ``n_oversamples`` and ``n_power_iters`` arguments.
    Note that ``pca`` without centering is equivalent to a truncated SVD.
  - ``hilbert`` has been introduced. It creates the famous ill-conditioned Hilbert matrix.
    The matrix is suitable to stress test decompositions.
  - The ``arange`` procedure has been introduced. It creates evenly spaced value within a specified range
    and step

Bug fixes:
  - ``gemm`` could crash when the result was column major

Breaking
  - In ``symeig``, the ``eigenvectors`` argument is now called ``return_eigenvectors``.
  - In ``symeig`` with slice, the new ``uplo`` precedes the slice argument.
  - pca input "nb_components" has been renamed "n_components".
  - pca output tuple used the names (results, components). It has been renamed to (scores, loadings).
  - A ``pca`` overload that projected a data matrix on already existing principal_axis
    was removed. Simply multiply the data matrix with the loadings instead.

Deprecation:
  - The syntax gemm(A, B, C) is now deprecated.
    Use explicit "gemm(1.0, A, B, 0.0, C)" instead.
    Arguably not zero-ing C could also be a reasonable default.

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
