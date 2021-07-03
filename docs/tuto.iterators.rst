===================
Tutorial: Iterators
===================

Tensors can be iterated in the proper order. Arraymancer provides:

-  ``items`` and ``pairs``. ``pairs`` returns the coordinates of the
   tensor.

.. code:: nim

    import ../arraymancer, sequtils

    let a = toSeq(1..24).toTensor.reshape(2,3,4)
    # Tensor[system.int] of shape "[2, 3, 4]" on backend "Cpu"
    #           0                      1
    # |1      2     3     4| |13    14    15    16|
    # |5      6     7     8| |17    18    19    20|
    # |9     10    11    12| |21    22    23    24|

    for v in a:
      echo v

    for coord, v in a:
      echo coord
      echo v
    # @[0, 0, 0]
    # 1
    # @[0, 0, 1]
    # 2
    # @[0, 0, 2]
    # 3
    # @[0, 0, 3]
    # 4
    # @[0, 1, 0]
    # 5
    # @[0, 1, 1]
    # 6
    # @[0, 1, 2]
    # 7
    # @[0, 1, 3]
    # 8
    # @[0, 2, 0]
    # 9
    # ...

For convenience a ``values`` closure iterator is available for iterator
chaining. ``values`` is equivalent to ``items``.

A ``mitems`` iterator is available to directly mutate elements while
iterating. An ``axis`` iterator is available to iterate along an axis.
