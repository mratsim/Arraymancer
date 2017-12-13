Arraymancer - A n-dimensional tensor (ndarray) library.
=======================================================

Arraymancer is a tensor (N-dimensional array) project in Nim. The main
focus is providing a fast and ergonomic CPU and GPU ndarray library on
which to build a scientific computing and in particular a deep learning
ecosystem.

The library is inspired by Numpy and PyTorch. The library provides
ergonomics very similar to Numpy, Julia and Matlab but is fully parallel
and significantly faster than those libraries. It is also faster than
C-based Torch.

Note: While Nim is compiled and does not offer an interactive REPL yet
(like Jupyter), it allows much faster prototyping than C++ due to
extremely fast compilation times. Arraymancer compiles in about 5
seconds on my dual-core MacBook.

Why Arraymancer
---------------

I’ve identified several issues I want to tackle with Arraymancer:

The Python community is struggling to bring Numpy up-to-speed
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  Numba JIT compiler
-  Dask delayed parallel computation graph
-  Cython to ease numerical computations in Python
-  Due to the GIL shared-memory parallelism (OpenMP) is not possible in
   pure Python
-  Use “vectorized operations” (i.e. don’t use for loops in Python)

Why not use in a single language with all the blocks to build the most
efficient scientific computing library with Python ergonomics.

OpenMP batteries included.

A researcher workflow is a fight against inefficiencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Researchers in a heavy scientific computing domain often have the
following workflow: Mathematica/Matlab/Python/R (prototyping) ->
C/C++/Fortran (speed, memory)

Why not use in a language as productive as Python and as fast as C? Code
once, and don’t spend months redoing the same thing at a lower level.

Tools available in labs are not available in production:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

-  Managing and deploying Python (2.7, 3.5, 3.6) and packages version in
   a robust manner requires devops-fu (virtualenv, Docker, …)
-  Python data science ecosystem does not run on embedded devices
   (Nvidia Tegra/drones) or mobile phones, especially preprocessing
   dependencies.

Nim is compiled, no need to worry about version conflicts and the whole
toolchain being in need of fixing due to Python version updates.
Furthermore, as long as your platform supports C, Arraymancer will run
on it from Raspberry Pi to mobile phones and drones.

Note: Arraymancer Cuda/Cudnn backend is shaping up and convolutional
neural nets are just around the corner.

Bridging the gap between deep learning research and production
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The deep learning frameworks are currently in two camps: - Research:
Theano, Tensorflow, Keras, Torch, PyTorch - Production: Caffe, Darknet,
(Tensorflow)

Putting a research model in production, on a drone or as a webservice
for example, is difficult: - Transforming a tuned research model (in
Python) to a usable Caffe or Darknet model (in C) is not trivial. ~PMML
is supposed to be the “common” XML description of ML models but is not
really supported by anyone.~  **Edit - Sept 7, 2017: Microsoft and
Facebook are announcing `Open Neural Network
Exchange <https://research.fb.com/facebook-and-microsoft-introduce-new-open-ecosystem-for-interchangeable-ai-frameworks/>`__**

Furthermore, Python preprocessing steps, unless using OpenCV, often
needs a custom implementation (think text/speech preprocessing on
phones).

-  Tensorflow is supposed to bridge the gap between research and
   production but its syntax and ergonomics are a pain to work with.
   It’s the same issue as researchers, “Prototype in Keras, and when you
   need low-level –> Tensorflow”.
-  Deployed models are static, there is no interface to add a new
   observation/training sample to any framework, the end goal being to
   use a model as a webservice.

So why Arraymancer ?
~~~~~~~~~~~~~~~~~~~~

All those pain points may seem like a huge undertaking however thanks to
the Nim language, we can have Arraymancer: - Be as fast as C -
Accelerated routines with Intel MKL/OpenBLAS or even NNPACK - Access to
CUDA and CuDNN and generate custom CUDA kernels on the fly via
metaprogramming. - A Python-like syntax with custom operators ``a * b``
for tensor multiplication instead of ``a.dot(b)`` (Numpy/Tensorflow) or
``a.mm(b)`` (Torch) - Numpy-like slicing ergonomics ``t[0..4, 2..10|2]``
- For everything that Nim doesn’t have yet, you can use Nim bindings to
C, C++, Objective-C or Javascript to bring it to Nim. Nim also has
unofficial Python->Nim and Nim->Python wrappers.

Future ambitions
----------------

Because apparently to be successful you need a vision, I would like
Arraymancer to be: - The go-to tool for Deep Learning video processing.
I.e. ``vid = load_video("./cats/youtube_cat_video.mkv")`` - Target
javascript, WebAssembly, Apple Metal, ARM devices, AMD Rocm, OpenCL, you
name it. - Target cryptominers FPGAs because they drove the price of
GPUs for honest deep-learners too high.

Support (Types, OS, Hardware)
-----------------------------

Arraymancer’s tensors supports arbitrary types (floats, strings, objects
…).

Arraymancer run anywhere you can compile C code. Linux, MacOS are
supported, Windows should work too as Appveyor (Continuous Integration
for Windows) never flash red. Optionally you can compile Arraymancer
with Cuda support.

Note: Arraymancer Tensors and CudaTensors are tensors in the machine
learning sense (multidimensional array) not in the mathematical sense
(describe transformation laws)

Limitations:
------------

EXPERIMENTAL: Arraymancer may summon Ragnarok and cause the heat death
of the Universe.

1. Display of 5-dimensional or more tensors is not implemented. (To be
   honest Christopher Nolan had the same issue in Interstellar)

Installation:
-------------

Nim is available in some Linux repositories and on Homebrew for macOS.

I however recommend installing Nim in your user profile via
```choosenim`` <https://github.com/dom96/choosenim>`__. Once choosenim
installed Nim, you can ``nimble install arraymancer`` which will pull
arraymancer and all its dependencies.

Full documentation
------------------

Detailed API is available on Arraymancer official
`documentation <https://mratsim.github.io/Arraymancer/>`__.

Features
--------

For now Arraymancer is still at the ndarray stage, however a `vision
package <https://github.com/edubart/arraymancer-vision>`__ and a `deep
learning demo <https://github.com/edubart/arraymancer-demos>`__ are
available with logistic regression and perceptron from scratch.

You can also check the `detailed
example <https://github.com/mratsim/Arraymancer/blob/master/examples/ex01_xor_perceptron_from_scratch.nim>`__
or
`benchmark <https://github.com/mratsim/Arraymancer/blob/master/benchmarks/ex01_xor.nim>`__
perceptron for a preview of Arraymancer deep learning usage.

Speed
~~~~~

Parallelism
^^^^^^^^^^^

Most operations in Arraymancer are parallelized through OpenMP including
linear algebra functions, universal functions, ``map``, ``reduce`` and
``fold`` based operations.

Parallel loop fusion - YOLO (You Only Loop Once)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Arraymancer provides several constructs for the YOLO™ paradigm (You Only
Loop Once).

A naïve logistic sigmoid implementation in Numpy would be:

.. code:: python

    import math

    def sigmoid(x):
      return 1 / (1 + math.exp(-x))

With Numpy broadcasting, all those operations would be done on whole
tensors using Numpy C implementation, pretty efficient?

Actually no, this would create lots of temporary and loops across the
data: - ``temp1 = -x`` - ``temp2 = math.exp(temp1)`` -
``temp3 = 1 + temp2`` - ``temp4 = 1 / temp3``

So you suddenly get a o(n^4) algorithm.

Arraymancer can do the same using the explicit broadcast operator ``./``
and ``.+``. (To avoid name conflict we change the logistic sigmoid name)

.. code:: nim

    import arraymancer

    def customSigmoid[T: SomeReal](t: Tensor[T]): Tensor[T] =
      result = 1 ./ (1 .+ exp(-t))

Well, unfortunately, the only thing we gain here is parallelism but we
still have 4 loops over the data implicitly. Another way would be to use
the loop fusion template ``map_inline``:

.. code:: nim

    import arraymancer

    def customSigmoid2[T: SomeReal](t: Tensor[T]): Tensor[T] =
      result = map_inline(t):
        1 / (1 + exp(-x))

Now in a single loop over ``t``, Arraymancer will do
``1 / (1 + exp(-x))`` for each x found. ``x`` is a shorthand for the
elements of the first tensor argument.

Here is another example with 3 tensors and element-wise fused
multiply-add ``C += A .* B``:

.. code:: nim

    import arraymancer

    def fusedMultiplyAdd[T: SomeNumber](c: var Tensor[T], a, b: Tensor[T]) =
      ## Implements C += A .* B, .* is the element-wise multiply
      apply3_inline(c, a, b):
        x += y * z

Since the tensor were given in order (c, a, b): - x corresponds to
elements of c - y to a - z to b

Today Arraymancer offers ``map_inline``, ``map2_inline``,
``apply_inline``, ``apply2_inline`` and ``apply3_inline``.

Those are also parallelized using OpenMP. In the future, this will be
generalized to N inputs.

Similarly, ``reduce_inline`` and ``fold_inline`` are offered for
parallel, custom, fused reductions operations.

Memory allocation
^^^^^^^^^^^^^^^^^

For most operations in machine linear, memory and cache is the
bottleneck, for example taking the log of a Tensor can use at most 20%
of your theoretical max CPU speed (in GFLOPS) while matrix
multiplication can use 70%-90%+ for the best implementations (MKL,
OpenBLAS).

In the log case, the processor gives a result faster than it can load
data into its cache. In the matrix multiplication case, each element of
a matrix can be reused several times before loading data again.

Arraymancer strives hard to limit memory allocation with the ``inline``
version of ``map``, ``apply``, ``reduce``, ``fold`` (``map_inline``,
``apply_inline``, ``reduce_inline``, ``fold_inline``) mentionned above
that avoids intermediate results.

Furthermore while Arraymancer uses copy on assignment by default, most
procedures have an ``unsafe`` equivalent that provides no-copy
operations like ``reshape`` and ``unsafeReshape``. Warning ⚠: Memory is
shared in that case, modifying one of these tensors will modify the
other.

Safe vs unsafe: copy vs view
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Compared to most frameworks, Arraymancer choose to be safe by default
but allows ``unsafe`` operations to optimize for speed and memory. The
tensor resulting from ``unsafe`` operations (no-copy operations) share
the underlying storage with the input tensor (also called views or
shallow copies). This is often a surprise for beginners.

In the future Arraymancer will leverage Nim compiler to automatically
detect when an original is not used and modified anymore to
automatically replace it by the ``unsafe`` equivalent.

For CudaTensors, operations are unsafe by default (including assignmnt
with ``=``) while waiting for further Nim optimizations for manually
managed memory. CudaTensors can be copied safely with ``.clone``

Tensors on CPU and on Cuda
~~~~~~~~~~~~~~~~~~~~~~~~~~

Tensors and CudaTensors do not have the same features implemented yet.
Also Cuda Tensors can only be float32 or float64 while Cpu Tensor can be
integers, string, boolean or any custom object.

Here is a comparative table, not that this feature set is developing
very rapidly.

------------------------------------------------- --------- ---------------------------------------------------------------
 Action                                           Tensor    CudaTensor
------------------------------------------------- --------- ---------------------------------------------------------------
 Accessing tensor properties                      [x]       [x]
 Tensor creation                                  [x]       by converting a cpu Tensor
 Accessing or modifying a single value            [x]       []
 Iterating on a Tensor                            [x]       []
 Slicing a Tensor                                 [x]       [x]
 Slice mutation ``a[1,_] = 10``                   [x]       []
 Comparison ``==``                                [x]       Coming soon
 Element-wise basic operations                    [x]       [x]
 Universal functions                              [x]       [x]
 Automatically broadcasted operations             [x]       Coming soon
 Matrix-Matrix and Matrix vector multiplication   [x]       [x] Note: sliced CudaTensors must explicitly be made contiguous
 Displaying a tensor                              [x]       [x]
 Higher-order functions (map, apply, reduce, fold)[x]       Apply, but only for internal use
 Transposing                                      [x]       [x]
 Converting to contiguous                         [x]       [x]
 Reshaping                                        [x]       []
 Explicit broadcast                               [x]       Coming soon
 Permuting dimensions                             [x]       Coming soon
 Concatenating along existing dimensions          [x]       []
 Squeezing singleton dimensions                   [x]       Coming soon
 Slicing + squeezing in one operation             [x]       Coming soon
------------------------------------------------- --------- ---------------------------------------------------------------
