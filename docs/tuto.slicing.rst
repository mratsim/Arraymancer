==============================
Tutorial: Slicing
==============================

Arraymancer supports the following slicing syntax. It allows for
selecting dimension subsets, whole dimension, stepping (one out of 2
rows), reversing dimensions, counting from the end.

.. code:: nim

    import math, arraymancer, future

    const
        x = @[1, 2, 3, 4, 5]
        y = @[1, 2, 3, 4, 5]

    var
        vandermonde: seq[seq[int]]
        row: seq[int]

    vandermonde = newSeq[seq[int]]()

    for i, xx in x:
        row = newSeq[int]()
        vandermonde.add(row)
        for j, yy in y:
            vandermonde[i].add(xx^yy)

    let foo = vandermonde.toTensor()

    echo foo

    # Tensor of shape 5x5 of type "int" on backend "Cpu"
    # |1      1       1       1       1|
    # |2      4       8       16      32|
    # |3      9       27      81      243|
    # |4      16      64      256     1024|
    # |5      25      125     625     3125|

    echo foo[1..2, 3..4] # slice

    # Tensor of shape 2x2 of type "int" on backend "Cpu"
    # |16     32|
    # |81     243|

    echo foo[3.._, _] # Span slice

    # Tensor of shape 2x5 of type "int" on backend "Cpu"
    # |4      16      64      256     1024|
    # |5      25      125     625     3125|

    echo foo[_..^3, _] # Slice until (inclusive, consistent with Nim)

    # Tensor of shape 3x5 of type "int" on backend "Cpu"
    # |1      1       1       1       1|
    # |2      4       8       16      32|
    # |3      9       27      81      243|

    echo foo[_.._|2, _] # Step

    # Tensor of shape 3x5 of type "int" on backend "Cpu"
    # |1      1       1       1       1|
    # |3      9       27      81      243|
    # |5      25      125     625     3125|

    echo foo[^1..0|-1, _] # Reverse step

    # Tensor of shape 5x5 of type "int" on backend "Cpu"
    # |5      25      125     625     3125|
    # |4      16      64      256     1024|
    # |3      9       27      81      243|
    # |2      4       8       16      32|
    # |1      1       1       1       1|

Slice mutations
~~~~~~~~~~~~~~~

Slices can also be mutated with a single value, a nested seq or array, a
tensor or tensor slice.

.. code:: nim

    import math, arraymancer, future

    const
        x = @[1, 2, 3, 4, 5]
        y = @[1, 2, 3, 4, 5]

    var
        vandermonde: seq[seq[int]]
        row: seq[int]

    vandermonde = newSeq[seq[int]]()

    for i, xx in x:
        row = newSeq[int]()
        vandermonde.add(row)
        for j, yy in y:
            vandermonde[i].add(xx^yy)

    var foo = vandermonde.toTensor()

    echo foo

    # Tensor of shape 5x5 of type "int" on backend "Cpu"
    # |1      1       1       1       1|
    # |2      4       8       16      32|
    # |3      9       27      81      243|
    # |4      16      64      256     1024|
    # |5      25      125     625     3125|

    # Mutation with a single value
    foo[1..2, 3..4] = 999

    echo foo
    # Tensor of shape 5x5 of type "int" on backend "Cpu"
    # |1      1       1       1       1|
    # |2      4       8       999     999|
    # |3      9       27      999     999|
    # |4      16      64      256     1024|
    # |5      25      125     625     3125|

    # Mutation with nested array or nested seq
    foo[0..1,0..1] = [[111, 222], [333, 444]]

    echo foo
    # Tensor of shape 5x5 of type "int" on backend "Cpu"
    # |111    222     1       1       1|
    # |333    444     8       999     999|
    # |3      9       27      999     999|
    # |4      16      64      256     1024|
    # |5      25      125     625     3125|

    # Mutation with a tensor or tensor slice.
    foo[^2..^1,2..4] = foo[^1..^2|-1, 4..2|-1]

    echo foo
    # Tensor of shape 5x5 of type "int" on backend "Cpu"
    # |111    222     1       1       1|
    # |333    444     8       999     999|
    # |3      9       27      999     999|
    # |4      16      3125    625     125|
    # |5      25      1024    256     64|