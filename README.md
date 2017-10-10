[![Join the chat at https://gitter.im/Arraymancer/Lobby](https://badges.gitter.im/Arraymancer/Lobby.svg)](https://gitter.im/Arraymancer/Lobby?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge) [![Linux Build Status (Travis)](https://img.shields.io/travis/mratsim/Arraymancer/master.svg?label=Linux%20/%20macOS "Linux/macOS build status (Travis)")](https://travis-ci.org/mratsim/Arraymancer) [![Windows build status (Appveyor)](https://img.shields.io/appveyor/ci/nicolargo/glances/master.svg?label=Windows "Windows build status (Appveyor)")](https://ci.appveyor.com/project/mratsim/arraymancer) [![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0) ![Stability](https://img.shields.io/badge/stability-experimental-orange.svg)

# Arraymancer - A n-dimensional tensor (ndarray) library

Arraymancer is a tensor (N-dimensional array) project. The main focus is providing a fast and ergonomic CPU and GPU ndarray library on which to build a numerical computing and in particular a deep learning ecosystem.

The library is inspired by Numpy and PyTorch.

<!-- TOC -->

- [Arraymancer - A n-dimensional tensor (ndarray) library](#arraymancer---a-n-dimensional-tensor-ndarray-library)
  - [Why Arraymancer](#why-arraymancer)
  - [Future ambitions](#future-ambitions)
  - [Support (Types, OS, Hardware)](#support-types-os-hardware)
  - [Limitations:](#limitations)
  - [Installation:](#installation)
  - [Full documentation](#full-documentation)
  - [Features](#features)
    - [Speed](#speed)
      - [Micro benchmark: Int64 matrix multiplication](#micro-benchmark-int64-matrix-multiplication)
      - [Logistic regression](#logistic-regression)
      - [DNN - 3 hidden layers](#dnn---3-hidden-layers)
    - [Safe vs unsafe: copy vs view](#safe-vs-unsafe-copy-vs-view)
    - [Tensors on CPU and on Cuda](#tensors-on-cpu-and-on-cuda)
    - [Tensor properties](#tensor-properties)
    - [Tensor creation](#tensor-creation)
    - [Accessing and modifying a value](#accessing-and-modifying-a-value)
    - [Copying](#copying)
    - [Slicing](#slicing)
    - [Slice mutations](#slice-mutations)
    - [Shapeshifting](#shapeshifting)
      - [Transposing](#transposing)
      - [Reshaping](#reshaping)
      - [Permuting - Reordering dimension](#permuting---reordering-dimension)
      - [Concatenation](#concatenation)
    - [Universal functions](#universal-functions)
    - [Type conversion](#type-conversion)
    - [Matrix and vector operations](#matrix-and-vector-operations)
    - [Broadcasting](#broadcasting)
    - [Iterators](#iterators)
    - [Higher-order functions (Map, Reduce, Fold)](#higher-order-functions-map-reduce-fold)
      - [`map`, `apply`, `map2`, `apply2`](#map-apply-map2-apply2)
      - [`reduce` on the whole Tensor or along an axis](#reduce-on-the-whole-tensor-or-along-an-axis)
      - [`fold` on the whole Tensor or along an axis](#fold-on-the-whole-tensor-or-along-an-axis)
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

All those pain points may seem like a huge undertaking however thanks to the Nim language, we can have Arraymancer:
- Be as fast as C
- Accelerated routines with Intel MKL/OpenBLAS or even NNPACK
- Access to CUDA and generate custom CUDA kernels on the fly via metaprogramming.
- A Python-like syntax with custom operators `a * b` for tensor multiplication instead of `a.dot(b)` (Numpy/Tensorflow) or `a.mm(b)` (Torch)
- Numpy-like slicing ergonomics `t[0..4, 2..10|2]`

## Future ambitions
Because apparently to be successful you need a vision, I would like Arraymancer to be:
- The go-to tool for Deep Learning video processing. I.e. `vid = load_video("./cats/youtube_cat_video.mkv")`
- Target javascript, WebAssembly, Apple Metal, ARM devices, AMD Rocm, OpenCL, you name it.
- Target cryptominers FPGAs because they drove the price of GPUs for honest deep-learners too high.

## Support (Types, OS, Hardware)
Arraymancer's tensors supports arbitrary types (floats, strings, objects ...).

Arraymancer run anywhere you can compile C code. Linux, MacOS are supported, Windows should work too as Appveyor (Continuous Integration for Windows) never flash red.
Optionally you can compile Arraymancer with Cuda support.

Note: Arraymancer Tensors and CudaTensors are tensors in the machine learning sense (multidimensional array) not in the mathematical sense (describe transformation laws)

## Limitations:

EXPERIMENTAL: Arraymancer may summon Ragnarok and cause the heat death of the Universe.

1. Display of 5-dimensional or more tensors is not implemented. (To be honest Christopher Nolan had the same issue in Interstellar)

## Installation:

Nim is available in some Linux repositories and on Homebrew for macOS.

I however recommend installing Nim in your user profile via [``choosenim``](https://github.com/dom96/choosenim). Once choosenim installed Nim, you can `nimble arraymancer` which will pull arraymancer and all its dependencies.

## Full documentation

Detailed API is available on Arraymancer official [documentation](https://mratsim.github.io/Arraymancer/).

## Features

For now Arraymancer is still at the ndarray stage, however a [vision package](https://github.com/edubart/arraymancer-vision) and a [deep learning demo](https://github.com/edubart/arraymancer-demos) are available with logistic regression and perceptron from scratch.

You can also check the [detailed example](https://github.com/mratsim/Arraymancer/blob/master/examples/ex01_xor_perceptron_from_scratch.nim) or [benchmark](https://github.com/mratsim/Arraymancer/blob/master/benchmarks/ex01_xor.nim) perceptron for a preview of Arraymancer deep learning usage.

### Speed

#### Micro benchmark: Int64 matrix multiplication

Integers seem to be the abandoned children of ndarrays and tensors libraries. Everyone is optimising the hell of floating points. Not so with Arraymancer:

```
Archlinux, E3-1230v5 (Skylake quad-core 3.4 GHz, turbo 3.8)
Input 1500x1500 random large int64 matrix
Arraymancer 0.2.90 (master branch 2017-10-10)
```

| Language | Speed | Memory |
|---|---|---|
| Nim 0.17.3 (devel) + OpenMP | **0.36s** | 55.5 MB |
| Julia v0.6.0 | 3.11s | 207.6 MB |
| Python 3.6.2 + Numpy 1.12 compiled from source | 8.03s | 58.9 MB |

```
MacOS + i5-5257U (Broadwell dual-core mobile 2.7GHz, turbo 3.1)
Input 1500x1500 random large int64 matrix
Arraymancer 0.2.90 (master branch 2017-10-10)
```

| Language | Speed | Memory |
|---|---|---|
| Nim 0.17.3 (devel) without OpenMP | **1.72s** | 90.0 MB |
| Julia v0.6.0 | 4.49s | 185.2 MB |
| Python 3.5.2 + Numpy 1.12 | 9.49s | 55.8 MB |

Benchmark setup is in the `./benchmarks` folder and similar to (stolen from) [Kostya's](https://github.com/kostya/benchmarks#matmul). Note: Arraymancer float matmul is as fast as `Julia Native Thread`.

#### Logistic regression
On the [demo benchmark](https://github.com/edubart/arraymancer-demos), Arraymancer is already faster than Torch in v0.2.0, further improvements in the "master" branch widened the gap since that benchmark.

| Framework | Backend | Forward+Backward Pass Time  |
|---|---|---|
| Arraymancer v0.2.0| OpenMP + MKL | **0.553ms**  |
| Torch7 | MKL | 0.733ms  |
| Arraymancer | OpenMP + OpenBLAS | 1.824ms |
| Numpy | MKL | 8.713ms  |

#### DNN - 3 hidden layers
| Framework | Backend | Forward+Backward Pass Time  |
|---|---|---|
| Arraymancer v0.2.0| OpenMP + MKL | **6.815ms**  |
| PyTorch | MKL | 7.320ms  |
| Arraymancer | OpenMP + OpenBLAS | 11.275ms |

```
Intel(R) Core(TM) i7-3770K CPU @ 3.50GHz, gcc 7.2.0, MKL 2017.17.0.4.4, OpenBLAS 0.2.20
```

In the future, Arraymancer will leverage Nim compiler to automatically fuse operations
like `alpha A*B + beta C` or a combination of element-wise operations. This is already done to fuse `toTensor` and `reshape`.

### Safe vs unsafe: copy vs view
Compared to most frameworks, Arraymancer choose to be safe by default but allows `unsafe` operations to optimize for speed and memory. The tensor resulting from `unsafe` operations (no-copy operations) share the underlying storage with the input tensor (also called views or shallow copies). This is often a surprise for beginners.

In the future Arraymancer will leverage Nim compiler to automatically detect when an original is not used and modified anymore to automatically replace it by the `unsafe` equivalent.

For CudaTensors, operations are unsafe by default (including assignmnt with `=`) while waiting for further Nim optimizations for manually managed memory. CudaTensors can be copied safely with `.clone`


### Tensors on CPU and on Cuda
Tensors and CudaTensors do not have the same features implemented yet.
Also Cuda Tensors can only be float32 or float64 while Cpu Tensor can be integers, string, boolean or any custom object.

Here is a comparative table, not that this feature set is developing very rapidly.

| Action | Tensor | CudaTensor |
| ------ | ------ | ---------- |
| Accessing tensor properties |[x]|[x]|
| Tensor creation |[x]| by converting a cpu Tensor|
| Accessing or modifying a single value |[x]|[]|
| Iterating on a Tensor |[x]|[]|
| Slicing a Tensor |[x]|[x]|
| Slice mutation `a[1,_] = 10` |[x]|[]|
| Comparison `==`|[x]| Coming soon|
| Element-wise basic operations|[x]|[x]|
| Universal functions |[x]|[x]|
| Automatically broadcasted operations |[x]| Coming soon|
| Matrix-Matrix and Matrix-Vector multiplication|[x]|[x] Note that sliced CudaTensors must explicitly be made contiguous for the moment|
| Displaying a tensor |[x]|[x]|
| Higher-order functions (map, apply, reduce, fold)|[x]| Apply, but only internally|
| Transposing | [x] | [x] |
| Converting to contiguous | [x] | [x] |
| Reshaping |[x] | [] |
| Explicit broadcast | [x] | Coming soon |
| Permuting dimensions | [x]| Coming soon |
| Concatenating tensors along existing dimension | [x]|[]|
| Squeezing singleton dimension |[x]| Coming soon|
| Slicing + squeezing |[x] | Coming soon |


### Tensor properties
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

let d = [[1, 2, 3], [4, 5, 6]].toTensor()

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

`toTensor` supports deep nested sequences and arrays, even sequence of arrays of sequences.

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
        ].toTensor()
echo c

# Tensor of shape 4x2x3 of type "int" on backend "Cpu"
#  |      1       2       3 |     11      22      33 |    111     222     333 |   1111    2222    3333|
#  |      4       5       6 |     44      55      66 |    444     555     666 |   4444    5555    6666|
```

`newTensor` procedure can be used to initialize a tensor of a specific shape with a default value. (0 for numbers, false for bool ...)

`zeros` and `ones` procedures create a new tensor filled with 0 and 1 respectively.

`zeros_like` and `ones_like` take an input tensor and output a tensor of the same shape but filled with 0 and 1 respectively.

```Nim
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
```

### Accessing and modifying a value

Tensors value can be retrieved or set with array brackets.

```Nim
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
```

### Copying

Tensor copy is deep by default (all the data is copied). In the majority of cases Nim compiler will detect and avoid useless copies.

`unsafeView` can be used on a Tensor to enforce shallow copying (data is shared between the 2 variables). Most shape manipulation proc also have an `unsafe` version.

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
```

### Shapeshifting

#### Transposing
The `transpose` function will reverse the dimensions of a tensor.

#### Reshaping
The `reshape` function will change the shape of a tensor. The number of elements in the new and old shape must be the same.

For example:
```Nim
let a = toSeq(1..24).toTensor().reshape(2,3,4)

# Tensor of shape 2x3x4 of type "int" on backend "Cpu"
#  |      1       2       3       4 |     13      14      15      16|
#  |      5       6       7       8 |     17      18      19      20|
#  |      9       10      11      12 |    21      22      23      24|
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

3 functions exist: `makeUniversal`, `makeUniversalLocal` and `map`.

`makeUniversal` create a a function that applies to each element of a tensor from any unary function. Most functions from the `math` module have been generalized to tensors with `makeUniversal(sin)`.
Furthermore those universal functions are exported and available for import.

`makeUniversalLocal` does not export the universal functions.

`map` is more generic and map any function to all element of a tensor. `map` works even if the function changes the type of the tensor's elements.

```Nim
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
```

### Type conversion

A type conversion fonction `astype` is provided for convenience
```Nim
let foo_float = foo.astype(float)
```

### Matrix and vector operations

The following linear algebra operations are supported for tensors of rank 1 (vectors) and 2 (matrices):

- dot product (Vector to Vector) using `dot`
- addition and substraction (any rank) using `+` and `-`
- in-place addition and substraction (any-rank) using `+=` and `-=`
- multiplication or division by a scalar using `*` and `/`
- matrix-matrix multiplication using `*`
- matrix-vector multiplication using `*`
- element-wise multiplication (Hadamard product) using `.*`

Note: Matrix operations for floats are accelerated using BLAS (Intel MKL, OpenBLAS, Apple Accelerate ...). Unfortunately there is no acceleration routine for integers. Integer matrix-matrix and matrix-vector multiplications are implemented via semi-optimized routines, see the [benchmarks section.](#micro-benchmark-int64-matrix-multiplication)

```Nim
echo foo_float * foo_float # Accelerated Matrix-Matrix multiplication (needs float)
# Tensor of shape 5x5 of type "float" on backend "Cpu"
# |15.0    55.0    225.0    979.0     4425.0|
# |258.0   1146.0  5274.0   24810.0   118458.0|
# |1641.0  7653.0  36363.0  174945.0  849171.0|
# |6372.0  30340.0 146244.0 710980.0  3478212.0|
# |18555.0 89355.0 434205.0 2123655.0 10436805.0|
```

### Broadcasting
Arraymancer supports explicit broadcasting with `broadcast` and its alias `bc`.
And supports implicit broadcasting with operations beginning with a dot:

Image from Scipy

![](https://scipy.github.io/old-wiki/pages/image004de9e.gif)

```Nim
let j = [0, 10, 20, 30].toTensor(Cpu).reshape(4,1)
let k = [0, 1, 2].toTensor(Cpu).reshape(1,3)

echo j .+ k
# Tensor of shape 4x3 of type "int" on backend "Cpu"
# |0      1       2|
# |10     11      12|
# |20     21      22|
# |30     31      32|
```

- `.+`,`.- `,
- `.*`: broadcasted element-wise matrix multiplication also called Hadamard product)
- `./`: broadcasted element-wise division or integer-division
- `.+=`, `.-=`, `.*=`, `./= `: in-place versions. Only the right operand is broadcastable.

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

### Higher-order functions (Map, Reduce, Fold)
Arraymancer supports efficient higher-order functions on the whole tensor or on an axis.
#### `map`, `apply`, `map2`, `apply2`
```Nim
a.map(x => x+1)
```
or
```Nim
proc plusone[T](x: T): T =
  x + 1
a.map(plusone) # Map the function plusone
```
Note: for basic operation, you can use implicit broadcasting instead `a .+ 1`

`apply` is the same as `map` but in-place.

`map2` and `apply2` takes 2 input tensors and respectively, return a new one or modify the first in-place.
```Nim
proc `**`[T](x, y: T): T = # We create a new power `**` function that works on 2 scalars
  pow(x, y)
a.map2(`**`, b)
# Or
map2(a, `**`, b)
```
#### `reduce` on the whole Tensor or along an axis
`reduce` apply a function like `+` or `max` on the whole Tensor[T] returning a single value T.

For example:
- Reducing with `+` returns the sum of all elements of teh Tensor.
- Reducing with `max` returns the biggest element of the Tensor

`reduce` can be applied along an axis, for example the sum along the rows of a Tensor.

#### `fold` on the whole Tensor or along an axis
`fold` is a generalization of `reduce`. Its starting value is not the first element of the Tensor.

It can do anything that reduce can, but also has other tricks because it is not constrained by the Tensor type or starting value.

For example:
- Reducing with `was_a_odd_and_what_about_b` and a starting value of `true` returns `true` if all elements are odd or `false` otherwise

Just in case
```Nim
proc was_a_odd_and_what_about_b[T: SomeInteger](a: bool, b: T): bool =
  return a and (b mod 2 == 1) # a is the result of previous computations, b is the new integer to check.
```


### Aggregate and Statistics

`sum` and `mean` functions are available to compute the sum and mean of a tensor.
`sum` and `mean` can also be computed along an axis with the `axis` argument.

Generic aggregates on the whole tensor or along an axis can be computed with `agg` and `agg_inplace` functions.
