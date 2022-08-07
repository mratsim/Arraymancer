[![Join the chat on Discord #nim-science](https://img.shields.io/discord/371759389889003530?color=blue&label=nim-science&logo=discord&logoColor=gold&style=flat-square)](https://discord.gg/f5hA9UK3dY) [![Github Actions CI](https://github.com/mratsim/arraymancer/workflows/Arraymancer%20CI/badge.svg)](https://github.com/mratsim/arraymancer/actions?query=workflow%3A%Arraymancer+CI%22+branch%3Amaster) [![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0) ![Stability](https://img.shields.io/badge/stability-experimental-orange.svg)

# Arraymancer - A n-dimensional tensor (ndarray) library.

Arraymancer is a tensor (N-dimensional array) project in Nim. The main focus is providing a fast and ergonomic CPU, Cuda and OpenCL ndarray library on which to build a scientific computing ecosystem.

The library is inspired by Numpy and PyTorch and targets the following use-cases:
  - N-dimensional arrays (tensors) for numerical computing
  - machine learning algorithms (as in Scikit-learn: least squares solvers, PCA and dimensionality reduction, classifiers, regressors and clustering algorithms, cross-validation).
  - deep learning

The ndarray component can be used without the machine learning and deep learning component.
It can also use the OpenMP, Cuda or OpenCL backends.

Note: While Nim is compiled and does not offer an interactive REPL yet (like Jupyter), it allows much faster prototyping than C++ due to extremely fast compilation times. Arraymancer compiles in about 5 seconds on my dual-core MacBook.

Reminder of supported compilation flags:
- `-d:release`: Nim release mode (no stacktraces and debugging information)
- `-d:danger`: No runtime checks like array bound checking
- `-d:openmp`: Multithreaded compilation
- `-d:mkl`: Use MKL, implies `openmp`
- `-d:openblas`: Use OpenBLAS
- by default Arraymancer will try to use your default `blas.so/blas.dll`
  Archlinux users may have to specify `-d:blas=cblas`.
  See [nimblas](https://github.com/unicredit/nimblas) for further configuration.
- `-d:cuda`: Build with Cuda support
- `-d:cudnn`: Build with CuDNN support, implies `cuda`.
- `-d:avx512`: Build with AVX512 support by supplying the
  `-mavx512dq` flag to gcc / clang. Without this flag the
  resulting binary does not use AVX512 even on CPUs that support
  it. Handing this flag however, makes the binary incompatible with
  CPUs that do *not* support it. See the comments in #505 for a
  discussion (from `v0.7.9`).
- You might want to tune library paths in [nim.cfg](nim.cfg) after installation for OpenBLAS, MKL and Cuda compilation.
  The current defaults should work on Mac and Linux.

## Show me some code

Arraymancer tutorial is available [here](https://mratsim.github.io/Arraymancer/tuto.first_steps.html).

Here is a preview of Arraymancer syntax.

### Tensor creation and slicing
```Nim
import math, arraymancer

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
```

### Reshaping and concatenation
```Nim
import arraymancer, sequtils


let a = toSeq(1..4).toTensor.reshape(2,2)

let b = toSeq(5..8).toTensor.reshape(2,2)

let c = toSeq(11..16).toTensor
let c0 = c.reshape(3,2)
let c1 = c.reshape(2,3)

echo concat(a,b,c0, axis = 0)
# Tensor of shape 7x2 of type "int" on backend "Cpu"
# |1      2|
# |3      4|
# |5      6|
# |7      8|
# |11     12|
# |13     14|
# |15     16|

echo concat(a,b,c1, axis = 1)
# Tensor of shape 2x7 of type "int" on backend "Cpu"
# |1      2       5       6       11      12      13|
# |3      4       7       8       14      15      16|
```

### Broadcasting

Image from Scipy

![](https://scipy.github.io/old-wiki/pages/image004de9e.gif)

```Nim
import arraymancer

let j = [0, 10, 20, 30].toTensor.reshape(4,1)
let k = [0, 1, 2].toTensor.reshape(1,3)

echo j +. k
# Tensor of shape 4x3 of type "int" on backend "Cpu"
# |0      1       2|
# |10     11      12|
# |20     21      22|
# |30     31      32|
```

### A simple two layers neural network

From [example 3](./examples/ex03_simple_two_layers.nim).

```Nim
import arraymancer, strformat

discard """
A fully-connected ReLU network with one hidden layer, trained to predict y from x
by minimizing squared Euclidean distance.
"""

# ##################################################################
# Environment variables

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
let (N, D_in, H, D_out) = (64, 1000, 100, 10)

# Create the autograd context that will hold the computational graph
let ctx = newContext Tensor[float32]

# Create random Tensors to hold inputs and outputs, and wrap them in Variables.
let
  x = ctx.variable(randomTensor[float32](N, D_in, 1'f32))
  y = randomTensor[float32](N, D_out, 1'f32)

# ##################################################################
# Define the model.

network TwoLayersNet:
  layers:
    fc1: Linear(D_in, H)
    fc2: Linear(H, D_out)
  forward x:
    x.fc1.relu.fc2

let
  model = ctx.init(TwoLayersNet)
  optim = model.optimizer(SGD, learning_rate = 1e-4'f32)

# ##################################################################
# Training

for t in 0 ..< 500:
  let
    y_pred = model.forward(x)
    loss = y_pred.mse_loss(y)

  echo &"Epoch {t}: loss {loss.value[0]}"

  loss.backprop()
  optim.update()
```

### Teaser A text generated with Arraymancer's recurrent neural network

From [example 6](./examples/ex06_shakespeare_generator.nim).

Trained 45 min on my laptop CPU on Shakespeare and producing 4000 characters

```
Whter!
Take's servant seal'd, making uponweed but rascally guess-boot,
Bare them be that been all ingal to me;
Your play to the see's wife the wrong-pars
With child of queer wretchless dreadful cold
Cursters will how your part? I prince!
This is time not in a without a tands:
You are but foul to this.
I talk and fellows break my revenges, so, and of the hisod
As you lords them or trues salt of the poort.

ROMEO:
Thou hast facted to keep thee, and am speak
Of them; she's murder'd of your galla?

# [...] See example 6 for full text generation samples
```

## Table of Contents
<!-- TOC -->

- [Arraymancer - A n-dimensional tensor (ndarray) library.](#arraymancer---a-n-dimensional-tensor-ndarray-library)
  - [Performance notice on Nim 0.20 & compilation flags](#performance-notice-on-nim-020--compilation-flags)
  - [Show me some code](#show-me-some-code)
    - [Tensor creation and slicing](#tensor-creation-and-slicing)
    - [Reshaping and concatenation](#reshaping-and-concatenation)
    - [Broadcasting](#broadcasting)
    - [A simple two layers neural network](#a-simple-two-layers-neural-network)
    - [Teaser A text generated with Arraymancer's recurrent neural network](#teaser-a-text-generated-with-arraymancers-recurrent-neural-network)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
  - [Full documentation](#full-documentation)
  - [Features](#features)
    - [Arraymancer as a Deep Learning library](#arraymancer-as-a-deep-learning-library)
      - [Fizzbuzz with fully-connected layers (also called Dense, Affine or Linear layers)](#fizzbuzz-with-fully-connected-layers-also-called-dense-affine-or-linear-layers)
      - [Handwritten digit recognition with convolutions](#handwritten-digit-recognition-with-convolutions)
      - [Sequence classification with stacked Recurrent Neural Networks](#sequence-classification-with-stacked-recurrent-neural-networks)
    - [Tensors on CPU, on Cuda and OpenCL](#tensors-on-cpu-on-cuda-and-opencl)
  - [What's new in Arraymancer v0.5.1 - July 2019](#whats-new-in-arraymancer-v051---july-2019)
  - [4 reasons why Arraymancer](#4-reasons-why-arraymancer)
    - [The Python community is struggling to bring Numpy up-to-speed](#the-python-community-is-struggling-to-bring-numpy-up-to-speed)
    - [A researcher workflow is a fight against inefficiencies](#a-researcher-workflow-is-a-fight-against-inefficiencies)
    - [Can be distributed almost dependency free](#can-be-distributed-almost-dependency-free)
    - [Bridging the gap between deep learning research and production](#bridging-the-gap-between-deep-learning-research-and-production)
    - [So why Arraymancer ?](#so-why-arraymancer-)
  - [Future ambitions](#future-ambitions)

<!-- /TOC -->

## Installation

Nim is available in some Linux repositories and on Homebrew for macOS.

I however recommend installing Nim in your user profile via [``choosenim``](https://github.com/dom96/choosenim). Once choosenim installed Nim, you can `nimble install arraymancer` which will pull the latest arraymancer release and all its dependencies.

To install Arraymancer development version you can use `nimble install arraymancer@#head`.

Arraymancer requires a BLAS and Lapack library.

- On Windows you can get [OpenBLAS ](https://github.com/xianyi/OpenBLAS/wiki/Precompiled-installation-packages)and [Lapack](https://icl.cs.utk.edu/lapack-for-windows/lapack/) for Windows.
- On MacOS, Apple Accelerate Framework is included in all MacOS versions and provides those.
- On Linux, you can download libopenblas and liblapack through your package manager.

## Full documentation

Detailed API is available at Arraymancer official [documentation](https://mratsim.github.io/Arraymancer/). Note: This documentation is only generated for 0.X release. Check the [examples folder](examples/) for the latest devel evolutions.

## Features

For now Arraymancer is mostly at the multidimensional array stage, in particular Arraymancer offers the following:

- Basic math operations generalized to tensors (sin, cos, ...)
- Matrix algebra primitives: Matrix-Matrix, Matrix-Vector multiplication.
- Easy and efficient slicing including with ranges and steps.
- No need to worry about "vectorized" operations.
- Broadcasting support. Unlike Numpy it is explicit, you just need to use `+.` instead of `+`.
- Plenty of reshaping operations: concat, reshape, split, chunk, permute, transpose.
- Supports tensors of up to 6 dimensions. For example a stack of 4 3D RGB minifilms of 10 seconds would be 6 dimensions:
  `[4, 10, 3, 64, 1920, 1080]` for `[nb_movies, time, colors, depth, height, width]`
- Can read and write .csv, Numpy (.npy) and HDF5 files.
- OpenCL and Cuda backed tensors (not as feature packed as CPU tensors at the moment).
- Covariance matrices.
- Eigenvalues and Eigenvectors decomposition.
- Least squares solver.
- K-means and PCA (Principal Component Analysis).

### Arraymancer as a Deep Learning library

Deep learning features can be explored but are considered unstable while I iron out their final interface.

Reminder: The final interface is still **work in progress.**

You can also watch the following animated [neural network demo](https://github.com/Vindaar/NeuralNetworkLiveDemo) which shows live training via [nim-plotly](https://github.com/brentp/nim-plotly).

#### Fizzbuzz with fully-connected layers (also called Dense, Affine or Linear layers)
Neural network definition extracted from [example 4](examples/ex04_fizzbuzz_interview_cheatsheet.nim).

```Nim
const
  NumDigits = 10
  NumHidden = 100

network FizzBuzzNet:
  layers:
    hidden: Linear(NumDigits, NumHidden)
    output: Linear(NumHidden, 4)
  forward x:
    x.hidden.relu.output

let
  ctx = newContext Tensor[float32]
  model = ctx.init(FizzBuzzNet)
  optim = model.optimizer(SGD, 0.05'f32)
# ....
echo answer
# @["1", "2", "fizz", "4", "buzz", "6", "7", "8", "fizz", "10",
#   "11", "12", "13", "14", "15", "16", "17", "fizz", "19", "buzz",
#   "fizz", "22", "23", "24", "buzz", "26", "fizz", "28", "29", "30",
#   "31", "32", "fizz", "34", "buzz", "36", "37", "38", "39", "40",
#   "41", "fizz", "43", "44", "fizzbuzz", "46", "47", "fizz", "49", "50",
#   "fizz", "52","53", "54", "buzz", "56", "fizz", "58", "59", "fizzbuzz",
#   "61", "62", "63", "64", "buzz", "fizz", "67", "68", "fizz", "buzz",
#   "71", "fizz", "73", "74", "75", "76", "77","fizz", "79", "buzz",
#   "fizz", "82", "83", "fizz", "buzz", "86", "fizz", "88", "89", "90",
#   "91", "92", "fizz", "94", "buzz", "fizz", "97", "98", "fizz", "buzz"]
```

#### Handwritten digit recognition with convolutions
Neural network definition extracted from [example 2](examples/ex02_handwritten_digits_recognition.nim).

```Nim
network DemoNet:
  layers:
    cv1:        Conv2D(@[1, 28, 28], out_channels = 20, kernel_size = (5, 5))
    mp1:        Maxpool2D(cv1.out_shape, kernel_size = (2,2), padding = (0,0), stride = (2,2))
    cv2:        Conv2D(mp1.out_shape, out_channels = 50, kernel_size = (5, 5))
    mp2:        MaxPool2D(cv2.out_shape, kernel_size = (2,2), padding = (0,0), stride = (2,2))
    fl:         Flatten(mp2.out_shape)
    hidden:     Linear(fl.out_shape[0], 500)
    classifier: Linear(500, 10)
  forward x:
    x.cv1.relu.mp1.cv2.relu.mp2.fl.hidden.relu.classifier

let
  ctx = newContext Tensor[float32] # Autograd/neural network graph
  model = ctx.init(DemoNet)
  optim = model.optimizer(SGD, learning_rate = 0.01'f32)

# ...
# Accuracy over 90% in a couple minutes on a laptop CPU
```

#### Sequence classification with stacked Recurrent Neural Networks
Neural network definition extracted [example 5](examples/ex05_sequence_classification_GRU.nim).

```Nim
const
  HiddenSize = 256
  Layers = 4
  BatchSize = 512


network TheGreatSequencer:
  layers:
    gru1: GRULayer(1, HiddenSize, 4) # (num_input_features, hidden_size, stacked_layers)
    fc1: Linear(HiddenSize, 32)                  # 1 classifier per GRU layer
    fc2: Linear(HiddenSize, 32)
    fc3: Linear(HiddenSize, 32)
    fc4: Linear(HiddenSize, 32)
    classifier: Linear(32 * 4, 3)                # Stacking a classifier which learns from the other 4
  forward x, hidden0:
    let
      (output, hiddenN) = gru1(x, hidden0)
      clf1 = hiddenN[0, _, _].squeeze(0).fc1.relu
      clf2 = hiddenN[1, _, _].squeeze(0).fc2.relu
      clf3 = hiddenN[2, _, _].squeeze(0).fc3.relu
      clf4 = hiddenN[3, _, _].squeeze(0).fc4.relu

    # Concat all
    # Since concat backprop is not implemented we cheat by stacking
    # Then flatten
    result = stack(clf1, clf2, clf3, clf4, axis = 2)
    result = classifier(result.flatten)

# Allocate the model
let
  ctx = newContext Tensor[float32]
  model = ctx.init(TheGreatSequencer)
  optim = model.optimizer(SGD, 0.01'f32)

# ...
let exam = ctx.variable([
    [float32 0.10, 0.20, 0.30], # increasing
    [float32 0.10, 0.90, 0.95], # increasing
    [float32 0.45, 0.50, 0.55], # increasing
    [float32 0.10, 0.30, 0.20], # non-monotonic
    [float32 0.20, 0.10, 0.30], # non-monotonic
    [float32 0.98, 0.97, 0.96], # decreasing
    [float32 0.12, 0.05, 0.01], # decreasing
    [float32 0.95, 0.05, 0.07]  # non-monotonic
# ...
echo answer.unsqueeze(1)
# Tensor[ex05_sequence_classification_GRU.SeqKind] of shape [8, 1] of type "SeqKind" on backend "Cpu"
# 	  Increasing|
# 	  Increasing|
# 	  Increasing|
# 	  NonMonotonic|
# 	  NonMonotonic|
# 	  Increasing| <----- Wrong!
# 	  Decreasing|
# 	  NonMonotonic|
```

#### Composing models
Network models can also act as layers in other network definitions.
The handwritten-digit-recognition model above can also be written like this:

```Nim

network SomeConvNet:
  layers h, w:
    cv1:        Conv2D(@[1, h, w], 20, (5, 5))
    mp1:        Maxpool2D(cv1.out_shape, (2,2), (0,0), (2,2))
    cv2:        Conv2D(mp1.out_shape, 50, (5, 5))
    mp2:        MaxPool2D(cv2.out_shape, (2,2), (0,0), (2,2))
    fl:         Flatten(mp2.out_shape)
  forward x:
    x.cv1.relu.mp1.cv2.relu.mp2.fl

# this model could be initialized like this: let model = ctx.init(SomeConvNet, h = 28, w = 28)

# functions `out_shape` and `in_shape` returning a `seq[int]` are convention (but not strictly necessary)
# for layers/models that have clearly defined output and input size
proc out_shape*[T](self: SomeConvNet[T]): seq[int] =
  self.fl.out_shape
proc in_shape*[T](self: SomeConvNet[T]): seq[int] =
  self.cv1.in_shape

network DemoNet:
  layers:
  # here we use the previously defined SomeConvNet as a layer
    cv:         SomeConvNet(28, 28)
    hidden:     Linear(cv.out_shape[0], 500)
    classifier: Linear(hidden.out_shape[0], 10)
  forward x:
    x.cv.hidden.relu.classifier
```

#### Custom layers
It is also possible to create fully custom layers.
The documentation for this can be found in the [official API documentation](https://mratsim.github.io/Arraymancer/nn_dsl.html).

### Tensors on CPU, on Cuda and OpenCL
Tensors, CudaTensors and CLTensors do not have the same features implemented yet.
Also CudaTensors and CLTensors can only be float32 or float64 while CpuTensors can be integers, string, boolean or any custom object.

Here is a comparative table of the core features.

| Action                                            | Tensor                      | CudaTensor                 | ClTensor                   |
| ------------------------------------------------- | --------------------------- | -------------------------- | -------------------------- |
| Accessing tensor properties                       | [x]                         | [x]                        | [x]                        |
| Tensor creation                                   | [x]                         | by converting a cpu Tensor | by converting a cpu Tensor |
| Accessing or modifying a single value             | [x]                         | []                         | []                         |
| Iterating on a Tensor                             | [x]                         | []                         | []                         |
| Slicing a Tensor                                  | [x]                         | [x]                        | [x]                        |
| Slice mutation `a[1,_] = 10`                      | [x]                         | []                         | []                         |
| Comparison `==`                                   | [x]                         | []                         | []                         |
| Element-wise basic operations                     | [x]                         | [x]                        | [x]                        |
| Universal functions                               | [x]                         | []                         | []                         |
| Automatically broadcasted operations              | [x]                         | [x]                        | [x]                        |
| Matrix-Matrix and Matrix-Vector multiplication    | [x]                         | [x]                        | [x]                        |
| Displaying a tensor                               | [x]                         | [x]                        | [x]                        |
| Higher-order functions (map, apply, reduce, fold) | [x]                         | internal only              | internal only              |
| Transposing                                       | [x]                         | [x]                        | []                         |
| Converting to contiguous                          | [x]                         | [x]                        | []                         |
| Reshaping                                         | [x]                         | [x]                        | []                         |
| Explicit broadcast                                | [x]                         | [x]                        | [x]                        |
| Permuting dimensions                              | [x]                         | []                         | []                         |
| Concatenating tensors along existing dimension    | [x]                         | []                         | []                         |
| Squeezing singleton dimension                     | [x]                         | [x]                        | []                         |
| Slicing + squeezing                               | [x]                         | []                         | []                         |

## What's new in Arraymancer v0.5.1 - July 2019

The full changelog is available in [changelog.md](./changelog.md).

Here are the highlights:
  - 0.20.x compatibility
  - Complex support
  - `Einsum`
  - Naive whitespace tokenizer for NLP
  - Fix height/width order when reading an image in tensor
  - Preview of Laser backend for matrix multiplication without SIMD autodetection (already 5x faster on integer matrix multiplication)

## 4 reasons why Arraymancer

### The Python community is struggling to bring Numpy up-to-speed

- Numba JIT compiler
- Dask delayed parallel computation graph
- Cython to ease numerical computations in Python
- Due to the GIL shared-memory parallelism (OpenMP) is not possible in pure Python
- Use "vectorized operations" (i.e. don't use for loops in Python)

Why not use in a single language with all the blocks to build the most efficient scientific computing library with Python ergonomics.

OpenMP batteries included.

### A researcher workflow is a fight against inefficiencies

Researchers in a heavy scientific computing domain often have the following workflow: Mathematica/Matlab/Python/R (prototyping) -> C/C++/Fortran (speed, memory)

Why not use in a language as productive as Python and as fast as C? Code once, and don't spend months redoing the same thing at a lower level.

### Can be distributed almost dependency free

Arraymancer models can be packaged in a self-contained binary that only depends on a BLAS library like OpenBLAS, MKL or Apple Accelerate (present on all Mac and iOS).

This means that there is no need to install a huge library or language ecosystem to use Arraymancer. This also makes it naturally suitable for resource-constrained devices like mobile phones and Raspberry Pi.

### Bridging the gap between deep learning research and production
The deep learning frameworks are currently in two camps:
- Research: Theano, Tensorflow, Keras, Torch, PyTorch
- Production: Caffe, Darknet, (Tensorflow)

Furthermore, Python preprocessing steps, unless using OpenCV, often needs a custom implementation (think text/speech preprocessing on phones).

- Managing and deploying Python (2.7, 3.5, 3.6) and packages version in a robust manner requires devops-fu (virtualenv, Docker, ...)
- Python data science ecosystem does not run on embedded devices (Nvidia Tegra/drones) or mobile phones, especially preprocessing dependencies.
- Tensorflow is supposed to bridge the gap between research and production but its syntax and ergonomics are a pain to work with. Like for researchers, you need to code twice, "Prototype in Keras, and when you need low-level --> Tensorflow".
- Deployed models are static, there is no interface to add a new observation/training sample to any framework, what if you want to use a model as a webservice with online learning?

[Relevant XKCD from Apr 30, 2018](https://xkcd.com/1987/)

![Python environment mess](https://imgs.xkcd.com/comics/python_environment.png)

### So why Arraymancer ?

All those pain points may seem like a huge undertaking however thanks to the Nim language, we can have Arraymancer:
- Be as fast as C
- Accelerated routines with Intel MKL/OpenBLAS or even NNPACK
- Access to CUDA and CuDNN and generate custom CUDA kernels on the fly via metaprogramming.
- Almost dependency free distribution (BLAS library)
- A Python-like syntax with custom operators `a * b` for tensor multiplication instead of `a.dot(b)` (Numpy/Tensorflow) or `a.mm(b)` (Torch)
- Numpy-like slicing ergonomics `t[0..4, 2..10|2]`
- For everything that Nim doesn't have yet, you can use Nim bindings to C, C++, Objective-C or Javascript to bring it to Nim. Nim also has unofficial Python->Nim and Nim->Python wrappers.

## Future ambitions
Because apparently to be successful you need a vision, I would like Arraymancer to be:
- The go-to tool for Deep Learning video processing. I.e. `vid = load_video("./cats/youtube_cat_video.mkv")`
- Target javascript, WebAssembly, Apple Metal, ARM devices, AMD Rocm, OpenCL, you name it.
- The base of a Starcraft II AI bot.
- Target cryptominers FPGAs because they drove the price of GPUs for honest deep-learners too high.
