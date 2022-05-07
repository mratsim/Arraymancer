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

    proc sigmoid(x):
      return 1 / (1 + math.exp(-x))

With Numpy broadcasting, all those operations would be done on whole
tensors using Numpy C implementation, pretty efficient?

Actually no, this would create lots of temporary and loops across the
data: - ``temp1 = -x`` - ``temp2 = math.exp(temp1)`` -
``temp3 = 1 + temp2`` - ``temp4 = 1 / temp3``

So you suddenly get a `O(4*n)` algorithm.

Arraymancer can do the same using the explicit broadcast operator ``/.``
and ``+.``. (To avoid name conflict we change the logistic sigmoid name)

.. code:: nim

    import arraymancer

    proc customSigmoid[T: SomeFloat](t: Tensor[T]): Tensor[T] =
      result = 1 /. (1 +. exp(-t))

Well, unfortunately, the only thing we gain here is parallelism but we
still have 4 loops over the data implicitly. Another way would be to use
the loop fusion template ``map_inline``:

.. code:: nim

    import arraymancer

    proc customSigmoid2[T: SomeFloat](t: Tensor[T]): Tensor[T] =
      result = map_inline(t):
        1 / (1 + exp(-x))

Now in a single loop over ``t``, Arraymancer will do
``1 / (1 + exp(-x))`` for each x found. ``x`` is a shorthand for the
elements of the first tensor argument.

Here is another example with 3 tensors and element-wise fused
multiply-add ``C += A *. B``:

.. code:: nim

    import arraymancer

    proc fusedMultiplyAdd[T: SomeNumber](c: var Tensor[T], a, b: Tensor[T]) =
      ## Implements C += A *. B, *. is the element-wise multiply
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

    Archlinux, i9-9980XE
    (Skylake-X 18 cores, overclocked 4.1GHz all-core turbo, 4.0GHz all-AVX-turbo, 3.5 GHz all-AVX512 turbo)
    Input 1500x1500 random large int64 matrix
    Arraymancer 81777f0 (~v0.6.0, master branch 2020-01-09)

--------------------------------------------------- ------------ ------------
 Language                                            Speed        Memory
--------------------------------------------------- ------------ ------------
 Nim 1.0.4 (-d:danger) + OpenMP                      **0.14s**     22.7 MB
 Julia v1.3.1                                         1.67s        246.5 MB
 Python 3.8.1 + Numpy-MKL 1.18 compiled from source   5.69s        75.9 MB
--------------------------------------------------- ------------ ------------

Benchmark setup is in the ``./benchmarks`` folder and similar to (stolen
from) `Kostya’s <https://github.com/kostya/benchmarks#matmul>`__.

Note:
Arraymancer, Julia and Numpy have the same speed as each other on **float** matrix multiplication
as they all use Assembly-based BLAS + OpenMP underneath.
In the future, pure-Nim backends without Assembly and/or OpenMP may be used
to ease deployment, especially on Windows and be free of OpenMP limitations
with regards to nested parallelism and `load-balancing of generic algorithms<https://github.com/zy97140/omp-benchmark-for-pytorch>`_.
Speed will be competitive at least with OpenBLAS, see the `Weave multithreading runtime benchmarks<https://github.com/mratsim/weave/tree/v0.4.0/benchmarks/matmul_gemm_blas>`_.
