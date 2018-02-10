=====================
Under the hood: Speed
=====================

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

For most operations in machine learning, memory and cache is the
bottleneck, for example taking the log of a Tensor can use at most 20%
of your theoretical max CPU speed (in GFLOPS) while matrix
multiplication can use 70%-90%+ for the best implementations (MKL,
OpenBLAS).

In the log case, the processor gives a result faster than it can load
data into its cache. In the matrix multiplication case, each element of
a matrix can be reused several times before loading data again.

Arraymancer strives hard to limit memory allocation with the ``inline``
version of ``map``, ``apply``, ``reduce``, ``fold`` (``map_inline``,
``apply_inline``, ``reduce_inline``, ``fold_inline``) mentioned above
that avoids intermediate results.

Micro benchmark: Int64 matrix multiplication
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Integers seem to be the abandoned children of ndarrays and tensors
libraries. Everyone is optimising the hell of floating points. Not so
with Arraymancer:

::

    Archlinux, E3-1230v5 (Skylake quad-core 3.4 GHz, turbo 3.8)
    Input 1500x1500 random large int64 matrix
    Arraymancer 0.2.90 (master branch 2017-10-10)

------------------------------------------------ ------------ ------------
 Language                                         Speed        Memory
------------------------------------------------ ------------ ------------
 Nim 0.17.3 (devel) + OpenMP                      **0.36s**    55.5 MB
 Julia v0.6.0                                     3.11s        207.6 MB
 Python 3.6.2 + Numpy 1.12 compiled from source   8.03s        58.9 MB
------------------------------------------------ ------------ ------------


::

    MacOS + i5-5257U (Broadwell dual-core mobile 2.7GHz, turbo 3.1)
    Input 1500x1500 random large int64 matrix
    Arraymancer 0.2.90 (master branch 2017-10-31)

    no OpenMP compilation: nim c -d:native -d:release --out:build/integer_matmul --nimcache:./nimcache benchmarks/integer_matmul.nim
    with OpenMP: nim c -d:openmp --cc:gcc --gcc.exe:"/usr/local/bin/gcc-6" --gcc.linkerexe:"/usr/local/bin/gcc-6"  -d:native -d:release --out:build/integer_matmul --nimcache:./nimcache benchmarks/integer_matmul.nim

------------------------------------------------ ------------ ------------
 Language                                         Speed        Memory
------------------------------------------------ ------------ ------------
 Nim 0.18.0 (devel) - GCC 6 + OpenMP              **0.95s**    71.9 MB
 Nim 0.18.0 (devel) - Apple Clang 9 - no OpenMP   1.73s        71.7 MB
 Julia v0.6.0                                     4.49s        185.2 MB
 Python 3.5.2 + Numpy 1.12                        9.49s        55.8 MB
------------------------------------------------ ------------ ------------

Benchmark setup is in the ``./benchmarks`` folder and similar to (stolen
from) `Kostya’s <https://github.com/kostya/benchmarks#matmul>`__. Note:
Arraymancer float matmul is as fast as ``Julia Native Thread``.

Logistic regression
^^^^^^^^^^^^^^^^^^^

On the `demo
benchmark <https://github.com/edubart/arraymancer-demos>`__, Arraymancer
is faster than Torch in v0.2.90.

CPU

-------------------- -------------- ----------------------------
 Framework            Backend        Forward+Backward Pass Time
-------------------- -------------- ----------------------------
 Arraymancer v0.3.0   OpenMP + MKL   **0.458ms**
 Torch7               MKL            0.686ms
 Numpy                MKL            0.723ms
-------------------- -------------- ----------------------------

GPU

-------------------- -------------- ----------------------------
 Framework            Backend        Forward+Backward Pass Time
-------------------- -------------- ----------------------------
 Arraymancer v0.3.0   Cuda            WIP
 Torch7               Cuda            0.286ms
-------------------- -------------- ----------------------------

DNN - 3 hidden layers
^^^^^^^^^^^^^^^^^^^^^

CPU

-------------------- -------------- ----------------------------
 Framework            Backend        Forward+Backward Pass Time
-------------------- -------------- ----------------------------
 Arraymancer v0.3.0   OpenMP + MKL   **2.907ms**
 PyTorch              MKL            6.797ms
-------------------- -------------- ----------------------------

GPU

-------------------- -------------- ----------------------------
 Framework            Backend        Forward+Backward Pass Time
-------------------- -------------- ----------------------------
 Arraymancer v0.3.0   Cuda           WIP
 PyTorch              Cuda           4.765ms
-------------------- -------------- ----------------------------

::

    Intel(R) Core(TM) i7-3770K CPU @ 3.50GHz, gcc 7.2.0, MKL 2017.17.0.4.4, OpenBLAS 0.2.20, Cuda 8.0.61, Geforce GTX 1080 Ti, Nim 0.18.0

In the future, Arraymancer will leverage Nim compiler to automatically
fuse operations like ``alpha A*B + beta C`` or a combination of
element-wise operations. This is already done to fuse ``toTensor`` and
``reshape``.