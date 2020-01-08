==============================
Tutorial: Matrix and vectors operations
==============================

The following linear algebra operations are supported for tensors of
rank 1 (vectors) and 2 (matrices):

-  dot product (Vector to Vector) using ``dot``
-  addition and substraction (any rank) using ``+`` and ``-``
-  in-place addition and substraction (any-rank) using ``+=`` and ``-=``
-  multiplication or division by a scalar using ``*`` and ``/``
-  matrix-matrix multiplication using ``*``
-  matrix-vector multiplication using ``*``
-  element-wise multiplication (Hadamard product) using ``*.``

Note: Matrix operations for floats are accelerated using BLAS (Intel
MKL, OpenBLAS, Apple Accelerate …). Unfortunately there is no
acceleration routine for integers. Integer matrix-matrix and
matrix-vector multiplications are implemented via semi-optimized
routines, see the `benchmarks
section. <#micro-benchmark-int64-matrix-multiplication>`__

.. code:: nim

    echo foo_float * foo_float # Accelerated Matrix-Matrix multiplication (needs float)
    # Tensor of shape 5x5 of type "float" on backend "Cpu"
    # |15.0    55.0    225.0    979.0     4425.0|
    # |258.0   1146.0  5274.0   24810.0   118458.0|
    # |1641.0  7653.0  36363.0  174945.0  849171.0|
    # |6372.0  30340.0 146244.0 710980.0  3478212.0|
    # |18555.0 89355.0 434205.0 2123655.0 10436805.0|
