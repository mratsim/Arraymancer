# Arraymancer

A tensor (N-dimensional array) project. Focus on machine learning, deep learning and numerical computing.

A tensor supports arbitrary types (floats, strings, objects ...).

EXPERIMENTAL: API may change and break.

Note: Machine learning tensors ARE NOT mathematical tensors.

## Goals

The automatic backpropagation library [Nim-RMAD](https://github.com/mratsim/nim-rmad) needs to be generalized to vectors and matrices, 3D, 4D, 5D tensors for deep learning.

This library aims to provided an efficient tensor/ndarray type. Focus will be on numerical computation (BLAS) and GPU support.
The library will be flexible enough to represent arbitrary N-dimensional arrays, especially for NLP word vectors.

## Current status

EXPERIMENTAL: Arraymancer may summon Ragnarok and cause the heat death of the Universe.

Arraymancer's tensors currently support the following:
* Wrapping any type: string, floats, object
* Getting and setting value at a specific index or slices. See the slicing examples for syntax.
* Creating a tensor from deep nested sequences or arrays (or arrays of seq of arrays but well ...)
* Universal functions (ufunc). Universal functions will apply on the whole Tensor. Functions from Nim math module are universal
  (ln, srqt, exp)
* Creating your own universal functions with `makeUniversal`, `makeUniversalLocal` and `fmap`.
    
    `fmap` can even be used on functions with input/ouput of different types.
* Optimized Linear Algebra through BLAS (via [nimblas](https://github.com/unicredit/nimblas)) for float32 and float64
  
  * Matrix-Matrix product (GEMM)
  * Matrix-Vector product (GEMV)
* Fast non-BLAS accelerated operations:

  * Tensor-Tensor addition, substraction
  * By scalar multiplication, addition, substraction and division

Limitations BLAS and Tensor-Tensor operations are **only available for Tensors** and transposed Tensors for now.
**Slices cannot be used for now.**

Check syntax examples in the test folder.

## Examples
```Nim
import math, ../arraymancer, future

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

let foo = vandermonde.toTensor(Cpu)

echo foo

# Tensor of shape 5x5 of type "int" on backend "Cpu"
# |1      1       1       1       1|
# |2      4       8       16      32|
# |3      9       27      81      243|
# |4      16      64      256     1024|
# |5      25      125     625     3125|

echo foo.fmap(x => x.isPowerOfTwo) # map a function (=> need import future)

# Tensor of shape 5x5 of type "bool" on backend "Cpu"
# |true   true    true    true    true|
# |true   true    true    true    true|
# |false  false   false   false   false|
# |true   true    true    true    true|
# |false  false   false   false   false|

let foo_float = foo.fmap(x => x.float)
echo ln foo_float # universal function (convert first to float for ln)

# Tensor of shape 5x5 of type "float" on backend "Cpu"
# |0.0    0.0     0.0     0.0     0.0|
# |0.6931471805599453     1.386294361119891       2.079441541679836       2.772588722239781       3.465735902799727|
# |1.09861228866811       2.19722457733622        3.295836866004329       4.394449154672439       5.493061443340548|
# |1.386294361119891      2.772588722239781       4.158883083359671       5.545177444479562       6.931471805599453|
# |1.6094379124341        3.218875824868201       4.828313737302302       6.437751649736401       8.047189562170502|

echo foo_float * foo_float # Accelerated Matrix-Matrix multiplication (needs float)
# Tensor of shape 5x5 of type "float" on backend "Cpu"
# |15.0    55.0    225.0    979.0     4425.0|
# |258.0   1146.0  5274.0   24810.0   118458.0|
# |1641.0  7653.0  36363.0  174945.0  849171.0|
# |6372.0  30340.0 146244.0 710980.0  3478212.0|
# |18555.0 89355.0 434205.0 2123655.0 10436805.0|


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
```

## Not prioritized

The following Numpy-like functionality:
* statistics (mean, median, stddev ...)

will be added on an as-needed basis.
