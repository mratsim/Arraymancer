=====================
Tutorial: First steps
=====================


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
    # Tensor[int] of shape "[2, 3]" on backend "Cpu"
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

    # Tensor[system.int] of shape "[4, 2, 3]" on backend "Cpu"
    #           0                      1                      2                      3
    # |1          2       3| |11        22      33| |111      222     333| |1111    2222    3333|
    # |4          5       6| |44        55      66| |444      555     666| |4444    5555    6666|

``newTensor`` procedure can be used to initialize a tensor of a specific
shape with a default value. (0 for numbers, false for bool …)

``zeros`` and ``ones`` procedures create a new tensor filled with 0 and
1 respectively.

``zeros_like`` and ``ones_like`` take an input tensor and output a
tensor of the same shape but filled with 0 and 1 respectively.

.. code:: nim

    let e = newTensor[bool]([2, 3])
    # Tensor[bool] of shape "[2, 3]" on backend "Cpu"
    # |false  false   false|
    # |false  false   false|

    let f = zeros[float]([4, 3])
    # Tensor[float] of shape "[4, 3]" on backend "Cpu"
    # |0.0    0.0     0.0|
    # |0.0    0.0     0.0|
    # |0.0    0.0     0.0|
    # |0.0    0.0     0.0|

    let g = ones[float]([4, 3])
    # Tensor[float] of shape "[4, 3]" on backend "Cpu"
    # |1.0    1.0     1.0|
    # |1.0    1.0     1.0|
    # |1.0    1.0     1.0|
    # |1.0    1.0     1.0|

    let tmp = [[1,2],[3,4]].toTensor()
    let h = tmp.zeros_like
    # Tensor[int] of shape "[2, 2]" on backend "Cpu"
    # |0      0|
    # |0      0|

    let i = tmp.ones_like
    # Tensor[int] of shape "[2, 2]" on backend "Cpu"
    # |1      1|
    # |1      1|

Accessing and modifying a value
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Tensors value can be retrieved or set with array brackets.

.. code:: nim

    var a = toSeq(1..24).toTensor().reshape(2,3,4)

    echo a
    # Tensor[system.int] of shape "[2, 3, 4]" on backend "Cpu"
    #           0                      1
    # |1      2     3     4| |13    14    15    16|
    # |5      6     7     8| |17    18    19    20|
    # |9     10    11    12| |21    22    23    24|

    echo a[1, 1, 1]
    # 18

    a[1, 1, 1] = 999
    echo a
    # Tensor[system.int] of shape "[2, 3, 4]" on backend "Cpu"
    #             0                          1
    # |1        2      3      4| |13      14     15     16|
    # |5        6      7      8| |17     999     19     20|
    # |9       10     11     12| |21      22     23     24|

Copying
~~~~~~~

Warning ⚠: When you do the following, both tensors ``a`` and ``b`` will share data.
Full copy must be explicitly requested via the ``clone`` function.

.. code:: nim
    let a = toSeq(1..24).toTensor().reshape(2,3,4)
    var b = a

Here modifying ``b`` WILL modify ``a``.
This behaviour is the same as Numpy and Julia,
reasons can be found in the following `under the hood article<https://mratsim.github.io/Arraymancer/uth.copy_semantics.html>`_.

Tensor printing
~~~~~~~~~~~~~~~

As already seen in the examples above, printing of tensors of
arbitrary dimensionality is supported. For dimensions larger than 2 we
need to pick a way to represent them on a 2D screen.

We pick a representation that is possibly the most "natural"
generalization of pretty printing up to 2 dimensions. Consider the
following:

- A scalar is of "even" rank (0) and is printed as a 1x1 grid.
- A vector (*odd* rank 1) is represented by a *row* of scalars. That
  is a stacking of dimension N - 1 along the horizontal axis.
- A matrix (*even* rank 2) is represented by *stacking* rows of
  vectors. That is we extend along the *vertical* axis of elements of
  dimension N - 1.

From here we continue along the pattern:
- Odd dimensions N are *horizontal* stacks of the pretty print of N - 1
- Even dimensions N are *vertical* stacks of the pretty print of N - 1

To help with visibility separators ``|`` and ``-`` are applied between
stacks of different dimensions.

This yields a final 2D table of numbers where the dimension
"increases" from outside to inside.

If we have a tensor of shape ``[2, 3, 4, 3, 2]`` the most "outer"
layer is the first ``2``. As it is an odd dimension, this rank will be
stacked horizontally. The next dimension ``3`` will be a stack in
vertical. Inside of that are ``4`` horizontal stacks again until we
reach the last two dimensions ``[3, 2]``, which are simply printed as
expected for a 2D tensor.

To help with readability, the *index* of each of these dimensions is
printed on the top (odd dimension) / left (even dimension) of the
layer.

Take a look at the printing result of the aforementioned shape and try
to understand the indexing shown on the top / right and how it relates
to the different dimensions:

.. code:: nim
    let t1 = toSeq(1..144).toTensor().reshape(2,3,4,3,2)
    # Tensor[system.int] of shape "[2, 3, 4, 3, 2]" on backend "Cpu"
    #                           0                            |                            1
    #        0            1            2            3        |         0            1            2            3
    #   |1        2| |7        8| |13      14| |19      20|  |    |73      74| |79      80| |85      86| |91      92|
    # 0 |3        4| |9       10| |15      16| |21      22|  |  0 |75      76| |81      82| |87      88| |93      94|
    #   |5        6| |11      12| |17      18| |23      24|  |    |77      78| |83      84| |89      90| |95      96|
    #   ---------------------------------------------------  |    ---------------------------------------------------
    #        0            1            2            3        |         0            1            2            3
    #   |25      26| |31      32| |37      38| |43      44|  |    |97      98| |103    104| |109    110| |115    116|
    # 1 |27      28| |33      34| |39      40| |45      46|  |  1 |99     100| |105    106| |111    112| |117    118|
    #   |29      30| |35      36| |41      42| |47      48|  |    |101    102| |107    108| |113    114| |119    120|
    #   ---------------------------------------------------  |    ---------------------------------------------------
    #        0            1            2            3        |         0            1            2            3
    #   |49      50| |55      56| |61      62| |67      68|  |    |121    122| |127    128| |133    134| |139    140|
    # 2 |51      52| |57      58| |63      64| |69      70|  |  2 |123    124| |129    130| |135    136| |141    142|
    #   |53      54| |59      60| |65      66| |71      72|  |    |125    126| |131    132| |137    138| |143    144|
    #   ---------------------------------------------------  |    ---------------------------------------------------
