[![Linux Build Status (Travis)](https://travis-ci.org/mratsim/Arraymancer.svg?branch=master "Linux build status (Travis)")](https://travis-ci.org/mratsim/Arraymancer)   [![Windows build status (Appveyor)](https://ci.appveyor.com/api/projects/status/github/mratsim/arraymancer?branch=master&svg=true "Windows build status (Appveyor)")](https://ci.appveyor.com/project/mratsim/arraymancer) [![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0) ![Stability](https://img.shields.io/badge/stability-experimental-orange.svg)

# Arraymancer - A n-dimensional tensor (ndarray) library

Arraymancer is a tensor (N-dimensional array) project. The main focus is providing a fast and ergonomic CPU and GPU ndarray library on which to build a numerical computing and in particular a deep learning ecosystem.

The library is inspired by Numpy and PyTorch.

<!-- TOC -->

- [Arraymancer - A n-dimensional tensor (ndarray) library](#arraymancer---a-n-dimensional-tensor-ndarray-library)
    - [Why Arraymancer](#why-arraymancer)
    - [Support (Types, OS, Hardware)](#support-types-os-hardware)
    - [Limitations:](#limitations)
    - [Features](#features)
        - [Tensor properties](#tensor-properties)
        - [Tensor creation](#tensor-creation)
        - [Accessing and modifying a value](#accessing-and-modifying-a-value)
        - [Copying](#copying)
        - [Slicing](#slicing)
        - [Slice mutations](#slice-mutations)
        - [Shapeshifting](#shapeshifting)
            - [Transposing](#transposing)
            - [Reshaping](#reshaping)
            - [Broadcasting](#broadcasting)
            - [Permuting - Reordering dimension](#permuting---reordering-dimension)
            - [Concatenation](#concatenation)
        - [Universal functions](#universal-functions)
        - [Type conversion](#type-conversion)
        - [Matrix and vector operations](#matrix-and-vector-operations)
        - [Iterators](#iterators)
        - [Aggregate and Statistics](#aggregate-and-statistics)

<!-- /TOC -->

## Why Arraymancer
The deep learning frameworks are currently in two camps:
- Research: Theano, Tensorflow, Keras, Torch, PyTorch
- Production: Caffe, Darknet, (Tensorflow)

Putting a research model in production, on a drone or as a webservice for example, is difficult:
- Managing Python versions and environment is hell
- Python data science ecosystem does not run on embedded devices (Nvidia Tegra/drones) or mobile phones
- Transforming a tuned research model (in Python) to a usable Caffe or Darknet model (in C) is almost impossible. PMML is supposed to be the "common" XML description of ML models but is not really supported by anyone.
**Edit - Sept 7, 2017: Microsoft and Facebook are announcing [Open Neural Network Exchange](https://research.fb.com/facebook-and-microsoft-introduce-new-open-ecosystem-for-interchangeable-ai-frameworks/)**
- Tensorflow is supposed to bridge the gap between research and production but its syntax and ergonomics are a pain to work with.
- Deployed models are static, there is no interface to add a new observation/training sample to any framework. The end goal is to use a model as a webservice.
- No framework are designed yet with javascript/WebAssembly in mind.

All those pain points may seem like a huge undertaking however thanks to the Nim language, we can have Arraymancer:
- Be as fast as C
- Accelerated routines with Intel MKL/OpenBLAS or even NNPACK
- Access to CUDA and reusing existing Torch, Tensorflow or Nervana Neon kernels
- A Python-like syntax with custom operators `a .* b` for tensor multiplication instead of `a.dot(b)` (Numpy/Tensorflow) or `a.mm(b)` (Torch) and Numpy-like slicing ergonomics `t[0..4, 2..10|2]`
- Target javascript and soon WebAssembly

## Support (Types, OS, Hardware)
Arraymancer's tensors supports arbitrary types (floats, strings, objects ...).

Arraymancer will target PC and embedded devices running:
- Windows, MacOS, Linux
- Javascript/WebAssembly browsers
- X86, X86_64, ARM, Nvidia GPU
Jetson TX1 and embedded devices with GPU are also a target.
Provided ROCm (RadeonOpenCompute) can successfully use CUDA code, AMDs GPU will also be supported.

Magma will be supported for simultaneous computation on CPU + CUDA GPUs.
Currently only CPU backends are working.

Note: Arraymancer tensors are tensors in the machine learning sense (multidimensional array) not in the mathematical sense (describe transformation laws)

## Limitations:

EXPERIMENTAL: Arraymancer may summon Ragnarok and cause the heat death of the Universe.

1. Display of 5-dimensional or more tensors is not implemented.

## Features

### Tensor properties
Properties are read-only.

Tensors have the following properties:
- `rank`:
    - 0 for scalar (unfortunately cannot be stored)
    - 1 for vector
    - 2 for matrices
    - N for N-dimension array
- `shape`: a sequence of the tensor dimensions along each axis.

Next properties are technical and there for completeness
- `strides`: a sequence of numbers of steps to get the next item along a dimension.
- `offset`: the first element of the tensor

```Nim
import arraymancer

let d = [[1, 2, 3], [4, 5, 6]].toTensor(Cpu)

echo d
# Tensor of shape 2x3 of type "int" on backend "Cpu"
# |1      2       3|
# |4      5       6|

echo d.rank # 2
echo d.shape # @[2, 3]
echo d.strides # @[3, 1] => Next row is 3 elements away in memory while next column is 1 element away.
echo d.offset # 0

```

### Tensor creation

The canonical way to initialize a tensor is by converting a seq of seq of ... or an array of array of ... into a tensor using `toTensor`.

`toTensor` takes the backend (CPU-only currently) as a parameter and supports deep nested sequences and arrays.

```Nim
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
        ].toTensor(Cpu)
echo c

# Tensor of shape 4x2x3 of type "int" on backend "Cpu"
#  |      1       2       3 |     11      22      33 |    111     222     333 |   1111    2222    3333|
#  |      4       5       6 |     44      55      66 |    444     555     666 |   4444    5555    6666|
```

`newTensor` procedure can be used to initialize a tensor of a specific shape with a default value. (0 for numbers, false for bool ...)

`zeros` and `ones` procedures create a new tensor filled with 0 and 1 respectively.

`zeros_like` and `ones_like` take an input tensor and output a tensor of the same shape but filled with 0 and 1 respectively.

```Nim
let e = newTensor([2, 3], bool, Cpu)
# Tensor of shape 2x3 of type "bool" on backend "Cpu"
# |false  false   false|
# |false  false   false|

let f = zeros([4, 3], float, Cpu)
# Tensor of shape 4x3 of type "float" on backend "Cpu"
# |0.0    0.0     0.0|
# |0.0    0.0     0.0|
# |0.0    0.0     0.0|
# |0.0    0.0     0.0|

let g = ones([4, 3], float, Cpu)
# Tensor of shape 4x3 of type "float" on backend "Cpu"
# |1.0    1.0     1.0|
# |1.0    1.0     1.0|
# |1.0    1.0     1.0|
# |1.0    1.0     1.0|

let tmp = [[1,2],[3,4]].toTensor(Cpu)
let h = tmp.zeros_like
# Tensor of shape 2x2 of type "int" on backend "Cpu"
# |0      0|
# |0      0|

let i = tmp.ones_like
# Tensor of shape 2x2 of type "int" on backend "Cpu"
# |1      1|
# |1      1|
```

### Accessing and modifying a value

Tensors value can be retrieved or set with array brackets.

```Nim
var a = toSeq(1..24).toTensor(Cpu).reshape(2,3,4)

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
```

### Copying

Tensor copy is deep by default (all the data is copied). In the majority of cases Nim compiler will detect and avoid useless copies.

`shallowCopy` can be used on a var Tensor to enforce shallow copying (data is shared between the 2 variables).

### Slicing

Arraymancer supports the following slicing syntax. It allows for selecting dimension subsets, whole dimension, stepping (one out of 2 rows), reversing dimensions, counting from the end.

```Nim
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

let foo = vandermonde.toTensor(Cpu)

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
```

### Slice mutations

Slices can also be mutated with a single value, a nested seq or array, a tensor or tensor slice.

```Nim
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

var foo = vandermonde.toTensor(Cpu)

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
```

### Shapeshifting

#### Transposing
The `transpose` function will reverse the dimensions of a tensor.

#### Reshaping
The `reshape` function will change the shape of a tensor. The number of elements in the new and old shape must be the same.

For example:
```Nim
let a = toSeq(1..24).toTensor(Cpu).reshape(2,3,4)

# Tensor of shape 2x3x4 of type "int" on backend "Cpu"
#  |      1       2       3       4 |     13      14      15      16|
#  |      5       6       7       8 |     17      18      19      20|
#  |      9       10      11      12 |    21      22      23      24|
```

#### Broadcasting
Arraymancer supports explicit broadcasting with `broadcast` and its alias `bc`.
A future aim is to use `bc` as an indicator to automatically tune the shape of both tensors to make them compatible.
To avoid silent bugs, broadcasting is not implicit like for Numpy.

Image from Scipy

![](https://scipy.github.io/old-wiki/pages/image004de9e.gif)

```Nim
let j = [0, 10, 20, 30].toTensor(Cpu).reshape(4,1)
let k = [0, 1, 2].toTensor(Cpu).reshape(1,3)

echo j.bc([4,3]) + k.bc([4,3])
# Tensor of shape 4x3 of type "int" on backend "Cpu"
# |0      1       2|
# |10     11      12|
# |20     21      22|
# |30     31      32|
```

#### Permuting - Reordering dimension
The `permute` proc can be used to reorder dimensions.
Input is a tensor and the new dimension order

```Nim
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
```

#### Concatenation
Tensors can be concatenated along an axis with the `concat` proc.
```Nim
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

```

### Universal functions

Functions that applies to a single element can work on a whole tensor similar to Numpy's universal functions.

3 functions exist: `makeUniversal`, `makeUniversalLocal` and `fmap`.

`makeUniversal` create a a function that applies to each element of a tensor from any unary function. Most functions from the `math` module have been generalized to tensors with `makeUniversal(sin)`.
Furthermore those universal functions are exported and available for import.

`makeUniversalLocal` does not export the universal functions.

`fmap` is more generic and map any function to all element of a tensor. `fmap` works even if the function changes the type of the tensor's elements.

```Nim
echo foo.fmap(x => x.isPowerOfTwo) # map a function (`=>` comes from the future module )

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
```

### Type conversion

A type conversion fonction `astype` is provided for convenience
```Nim
let foo_float = foo.astype(float)
```

### Matrix and vector operations

The following linear algebra operations are supported for tensors of rank 1 (vectors) and 2 (matrices):

- dot product (Vector to Vector) using `.*`
- addition and substraction (any rank) using `+` and `-`
- in-place addition and substraction (any-rank) using `+=` and `-=`
- multiplication or division by a scalar using `*` and `/`
- matrix-matrix multiplication using `*`
- matrix-vector multiplication using `*`
- element-wise multiplication (Hadamard product) using `|*|`

Note: Matrix operations for floats are accelerated using BLAS (Intel MKL, OpenBLAS, Apple Accelerate ...). Unfortunately there is no acceleration routine for integers. Integer matrix-matrix and matrix-vector multiplications are implemented via semi-optimized routines (no naive loops but don't leverage CPU-specific features).

```Nim
echo foo_float * foo_float # Accelerated Matrix-Matrix multiplication (needs float)
# Tensor of shape 5x5 of type "float" on backend "Cpu"
# |15.0    55.0    225.0    979.0     4425.0|
# |258.0   1146.0  5274.0   24810.0   118458.0|
# |1641.0  7653.0  36363.0  174945.0  849171.0|
# |6372.0  30340.0 146244.0 710980.0  3478212.0|
# |18555.0 89355.0 434205.0 2123655.0 10436805.0|
```

### Iterators

Tensors can be iterated in the proper order. Arraymancer provides:

- `items` and `pairs`. `pairs` returns the coordinates of the tensor.

```Nim
import ../arraymancer, sequtils

let a = toSeq(1..24).toTensor(Cpu).reshape(2,3,4)
# Tensor of shape 2x3x4 of type "int" on backend "Cpu"
#  |      1       2       3       4 |     13      14      15      16|
#  |      5       6       7       8 |     17      18      19      20|
#  |      9       10      11      12 |    21      22      23      24|

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
```

For convenience a `values` closure iterator is available for iterator chaining. `values` is equivalent to `items`.

A `mitems` iterator is available to directly mutate elements while iterating.
An `axis` iterator is available to iterate along an axis.

### Aggregate and Statistics

`sum` and `mean` functions are avalaible to compute the sum and mean of a tensor.
`sum` and `mean` can also be computed along an axis with the `axis` argument.

Generic aggregates on the whole tensor or along an axis can be computed with `agg` and `agg_inplace` functions.
