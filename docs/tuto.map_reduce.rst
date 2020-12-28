====================================================
Tutorial: Higher-order functions (Map, Reduce, Fold)
====================================================


Arraymancer supports efficient higher-order functions on the whole
tensor or on an axis.

``map``, ``apply``, ``map2``, ``apply2``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: nim

    a.map(x => x+1)

or

.. code:: nim

    proc plusone[T](x: T): T =
      x + 1
    a.map(plusone) # Map the function plusone

Note: for basic operation, you can use implicit broadcasting instead
``a +. 1``

``apply`` is the same as ``map`` but in-place.

``map2`` and ``apply2`` takes 2 input tensors and respectively, return a
new one or modify the first in-place.

.. code:: nim

    proc `**`[T](x, y: T): T = # We create a new power `**` function that works on 2 scalars
      pow(x, y)
    a.map2(`**`, b)
    # Or
    map2(a, `**`, b)

``reduce`` on the whole Tensor or along an axis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``reduce`` apply a function like ``+`` or ``max`` on the whole Tensor[T]
returning a single value T.

For example: - Reducing with ``+`` returns the sum of all elements of
the Tensor. - Reducing with ``max`` returns the biggest element of the
Tensor

``reduce`` can be applied along an axis, for example the sum along the
rows of a Tensor.

``fold`` on the whole Tensor or along an axis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``fold`` is a generalization of ``reduce``. Its starting value is not
the first element of the Tensor.

It can do anything that reduce can, but also has other tricks because it
is not constrained by the Tensor type or starting value.

For example: - Reducing with ``was_a_odd_and_what_about_b`` and a
starting value of ``true`` returns ``true`` if all elements are odd or
``false`` otherwise

Just in case

.. code:: nim

    proc was_a_odd_and_what_about_b[T: SomeInteger](a: bool, b: T): bool =
      return a and (b mod 2 == 1) # a is the result of previous computations, b is the new integer to check.
