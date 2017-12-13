==============================
Spellbook: How to create an universal function?
==============================

Functions that applies to a single element can work on a whole tensor
similar to Numpy’s universal functions.

3 functions exist: ``makeUniversal``, ``makeUniversalLocal`` and
``map``.

``makeUniversal`` create a a function that applies to each element of a
tensor from any unary function. Most functions from the ``math`` module
have been generalized to tensors with ``makeUniversal(sin)``.
Furthermore those universal functions are exported and available for
import.

``makeUniversalLocal`` does not export the universal functions.

``map`` is more generic and map any function to all element of a tensor.
``map`` works even if the function changes the type of the tensor’s
elements.

.. code:: nim

    echo foo.map(x => x.isPowerOfTwo) # map a function (`=>` comes from the future module )

    # Tensor of shape 5x5 of type "bool" on backend "Cpu"
    # |true   true    true    true    true|
    # |true   true    true    true    true|
    # |false  false   false   false   false|
    # |true   true    true    true    true|
    # |false  false   false   false   false|

    let foo_float = foo.map(x => x.float)
    echo ln foo_float # universal function (convert first to float for ln)

    # Tensor of shape 5x5 of type "float" on backend "Cpu"
    # |0.0    0.0     0.0     0.0     0.0|
    # |0.6931471805599453     1.386294361119891       2.079441541679836       2.772588722239781       3.465735902799727|
    # |1.09861228866811       2.19722457733622        3.295836866004329       4.394449154672439       5.493061443340548|
    # |1.386294361119891      2.772588722239781       4.158883083359671       5.545177444479562       6.931471805599453|
    # |1.6094379124341        3.218875824868201       4.828313737302302       6.437751649736401       8.047189562170502|
