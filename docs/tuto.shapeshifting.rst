==============================
Tutorial: Transposing, Reshaping, Permuting, Concatenating
==============================


Shapeshifting
~~~~~~~~~~~~~

Transposing
^^^^^^^^^^^

The ``transpose`` function will reverse the dimensions of a tensor.

Reshaping
^^^^^^^^^

The ``reshape`` function will change the shape of a tensor. The number
of elements in the new and old shape must be the same.

For example:

.. code:: nim

    let a = toSeq(1..24).toTensor().reshape(2,3,4)

    # Tensor of shape 2x3x4 of type "int" on backend "Cpu"
    #  |      1       2       3       4 |     13      14      15      16|
    #  |      5       6       7       8 |     17      18      19      20|
    #  |      9       10      11      12 |    21      22      23      24|

Permuting - Reordering dimension
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``permute`` proc can be used to reorder dimensions. Input is a
tensor and the new dimension order

.. code:: nim

    let a = toSeq(1..24).toTensor(Cpu).reshape(2,3,4)
    echo a

    # Tensor of shape 2x3x4 of type "int" on backend "Cpu"
    #  |      1       2       3       4 |     13      14      15      16|
    #  |      5       6       7       8 |     17      18      19      20|
    #  |      9       10      11      12 |    21      22      23      24|

    echo a.permute(0,2,1) # dim 0 stays at 0, dim 1 becomes dim 2 and dim 2 becomes dim 1

    # Tensor of shape 2x4x3 of type "int" on backend "Cpu"
    #  |      1       5       9 |     13      17      21|
    #  |      2       6       10 |    14      18      22|
    #  |      3       7       11 |    15      19      23|
    #  |      4       8       12 |    16      20      24|

Concatenation
^^^^^^^^^^^^^

Tensors can be concatenated along an axis with the ``concat`` proc.

.. code:: nim

    import ../arraymancer, sequtils


    let a = toSeq(1..4).toTensor(Cpu).reshape(2,2)

    let b = toSeq(5..8).toTensor(Cpu).reshape(2,2)

    let c = toSeq(11..16).toTensor(Cpu)
    let c0 = c.reshape(3,2)
    let c1 = c.reshape(2,3)

    echo concat(a,b,c0, axis = 0)
    # Tensor of shape 7x2 of type "int" on backend "Cpu"
    # |1      2|
    # |3      4|
    # |5      6|
    # |7      8|
    # |11     12|
    # |13     14|
    # |15     16|

    echo concat(a,b,c1, axis = 1)
    # Tensor of shape 2x7 of type "int" on backend "Cpu"
    # |1      2       5       6       11      12      13|
    # |3      4       7       8       14      15      16|