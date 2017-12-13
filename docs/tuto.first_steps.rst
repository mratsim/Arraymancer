==============================
Tutorial: First steps
==============================


Tensor properties
~~~~~~~~~~~~~~~~~

Tensors have the following properties: - ``rank``: - 0 for scalar
(unfortunately cannot be stored) - 1 for vector - 2 for matrices - N for
N-dimension array - ``shape``: a sequence of the tensor dimensions along
each axis.

Next properties are technical and there for completeness - ``strides``:
a sequence of numbers of steps to get the next item along a dimension. -
``offset``: the first element of the tensor

.. code:: nim

    import arraymancer

    let d = [[1, 2, 3], [4, 5, 6]].toTensor()

    echo d
    # Tensor of shape 2x3 of type "int" on backend "Cpu"
    # |1      2       3|
    # |4      5       6|

    echo d.rank # 2
    echo d.shape # @[2, 3]
    echo d.strides # @[3, 1] => Next row is 3 elements away in memory while next column is 1 element away.
    echo d.offset # 0

Tensor creation
~~~~~~~~~~~~~~~

The canonical way to initialize a tensor is by converting a seq of seq
of … or an array of array of … into a tensor using ``toTensor``.

``toTensor`` supports deep nested sequences and arrays, even sequence of
arrays of sequences.

.. code:: nim

    import arraymancer

    let c = [
              [
                [1,2,3],
                [4,5,6]
              ],
              [
                [11,22,33],
                [44,55,66]
              ],
              [
                [111,222,333],
                [444,555,666]
              ],
              [
                [1111,2222,3333],
                [4444,5555,6666]
              ]
            ].toTensor()
    echo c

    # Tensor of shape 4x2x3 of type "int" on backend "Cpu"
    #  |      1       2       3 |     11      22      33 |    111     222     333 |   1111    2222    3333|
    #  |      4       5       6 |     44      55      66 |    444     555     666 |   4444    5555    6666|

``newTensor`` procedure can be used to initialize a tensor of a specific
shape with a default value. (0 for numbers, false for bool …)

``zeros`` and ``ones`` procedures create a new tensor filled with 0 and
1 respectively.

``zeros_like`` and ``ones_like`` take an input tensor and output a
tensor of the same shape but filled with 0 and 1 respectively.

.. code:: nim

    let e = newTensor([2, 3], bool)
    # Tensor of shape 2x3 of type "bool" on backend "Cpu"
    # |false  false   false|
    # |false  false   false|

    let f = zeros([4, 3], float)
    # Tensor of shape 4x3 of type "float" on backend "Cpu"
    # |0.0    0.0     0.0|
    # |0.0    0.0     0.0|
    # |0.0    0.0     0.0|
    # |0.0    0.0     0.0|

    let g = ones([4, 3], float)
    # Tensor of shape 4x3 of type "float" on backend "Cpu"
    # |1.0    1.0     1.0|
    # |1.0    1.0     1.0|
    # |1.0    1.0     1.0|
    # |1.0    1.0     1.0|

    let tmp = [[1,2],[3,4]].toTensor()
    let h = tmp.zeros_like
    # Tensor of shape 2x2 of type "int" on backend "Cpu"
    # |0      0|
    # |0      0|

    let i = tmp.ones_like
    # Tensor of shape 2x2 of type "int" on backend "Cpu"
    # |1      1|
    # |1      1|

Accessing and modifying a value
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Tensors value can be retrieved or set with array brackets.

.. code:: nim

    var a = toSeq(1..24).toTensor().reshape(2,3,4)

    echo a
    # Tensor of shape 2x3x4 of type "int" on backend "Cpu"
    #  |      1       2       3       4 |     13      14      15      16|
    #  |      5       6       7       8 |     17      18      19      20|
    #  |      9       10      11      12 |    21      22      23      24|

    echo a[1, 1, 1]
    # 18

    a[1, 1, 1] = 999
    echo a
    # Tensor of shape 2x3x4 of type "int" on backend "Cpu"
    #  |      1       2       3       4 |     13      14      15      16|
    #  |      5       6       7       8 |     17      999     19      20|
    #  |      9       10      11      12 |    21      22      23      24|

Copying
~~~~~~~

Tensor copy is deep by default (all the data is copied). In the majority
of cases Nim compiler will detect and avoid useless copies.

``unsafeView`` can be used on a Tensor to enforce shallow copying (data
is shared between the 2 variables). Most shape manipulation proc also
have an ``unsafe`` version.