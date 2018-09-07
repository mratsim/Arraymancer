[![Join the chat at https://gitter.im/Arraymancer/Lobby](https://badges.gitter.im/Arraymancer/Lobby.svg)](https://gitter.im/Arraymancer/Lobby?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge) [![Linux Build Status (Travis)](https://img.shields.io/travis/mratsim/Arraymancer/master.svg?label=Linux%20/%20macOS "Linux/macOS build status (Travis)")](https://travis-ci.org/mratsim/Arraymancer) [![Windows build status (Appveyor)](https://img.shields.io/appveyor/ci/mratsim/arraymancer/master.svg?label=Windows "Windows build status (Appveyor)")](https://ci.appveyor.com/project/mratsim/arraymancer) [![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0) ![Stability](https://img.shields.io/badge/stability-experimental-orange.svg)

# Arraymancer - A n-dimensional tensor (ndarray) library.

Arraymancer is a tensor (N-dimensional array) project in Nim. The main focus is providing a fast and ergonomic CPU, Cuda and OpenCL ndarray library on which to build a scientific computing and in particular a deep learning ecosystem.

The library is inspired by Numpy and PyTorch. The library provides ergonomics very similar to Numpy, Julia and Matlab but is fully parallel and significantly faster than those libraries. It is also faster than C-based Torch.

Note: While Nim is compiled and does not offer an interactive REPL yet (like Jupyter), it allows much faster prototyping than C++ due to extremely fast compilation times. Arraymancer compiles in about 5 seconds on my dual-core MacBook.

## Show me some code

Arraymancer tutorial is available [here](https://mratsim.github.io/Arraymancer/tuto.first_steps.html).

Here is a preview of Arraymancer syntax.

### Tensor creation and slicing
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

echo j .+ k
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

network ctx, TwoLayersNet:
  layers:
    fc1: Linear(D_in, H)
    fc2: Linear(H, D_out)
  forward x:
    x.fc1.relu.fc2

let
  model = ctx.init(TwoLayersNet)
  optim = model.optimizerSGD(learning_rate = 1e-4'f32)

# ##################################################################
# Training

for t in 0 ..< 500:
  let
    y_pred = model.forward(x)
    loss = mse_loss(y_pred, y)

  echo &"Epoch {t}: loss {loss.value[0]}"

  loss.backprop()
  optim.update()
```

## Table of Contents
<!-- TOC -->

- [Arraymancer - A n-dimensional tensor (ndarray) library.](#arraymancer---a-n-dimensional-tensor-ndarray-library)
  - [Show me some code](#show-me-some-code)
    - [Tensor creation and slicing](#tensor-creation-and-slicing)
    - [Reshaping and concatenation](#reshaping-and-concatenation)
    - [Broadcasting](#broadcasting)
    - [A simple two layers neural network](#a-simple-two-layers-neural-network)
  - [Table of Contents](#table-of-contents)
  - [4 reasons why Arraymancer](#4-reasons-why-arraymancer)
    - [The Python community is struggling to bring Numpy up-to-speed](#the-python-community-is-struggling-to-bring-numpy-up-to-speed)
    - [A researcher workflow is a fight against inefficiencies](#a-researcher-workflow-is-a-fight-against-inefficiencies)
    - [Can be distributed almost dependency free](#can-be-distributed-almost-dependency-free)
    - [Bridging the gap between deep learning research and production](#bridging-the-gap-between-deep-learning-research-and-production)
    - [So why Arraymancer ?](#so-why-arraymancer)
  - [Future ambitions](#future-ambitions)
  - [Installation](#installation)
  - [Full documentation](#full-documentation)
  - [Features](#features)
    - [Arraymancer as a Deep Learning library](#arraymancer-as-a-deep-learning-library)
      - [Handwritten digit recognition with Arraymancer](#handwritten-digit-recognition-with-arraymancer)
    - [Tensors on CPU, on Cuda and OpenCL](#tensors-on-cpu-on-cuda-and-opencl)
    - [Speed](#speed)
      - [Micro benchmark: Int64 matrix multiplication](#micro-benchmark-int64-matrix-multiplication)
      - [Logistic regression](#logistic-regression)
      - [DNN - 3 hidden layers](#dnn---3-hidden-layers)

<!-- /TOC -->

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

## Installation

Nim is available in some Linux repositories and on Homebrew for macOS.

I however recommend installing Nim in your user profile via [``choosenim``](https://github.com/dom96/choosenim). Once choosenim installed Nim, you can `nimble install arraymancer` which will pull the latest arraymancer release and all its dependencies.

To install Arraymancer development version you can use `nimble install arraymancer@#head`.

Arraymancer requires a BLAS and Lapack library.

- On Windows you can get OpenBLAS and Lapack for Windows.
- On MacOS, Apple Accelerate should provides those by default.
- On Linux, you can downlod libopenblas and liblapack through your package manager.

## Full documentation

Detailed API is available at Arraymancer official [documentation](https://mratsim.github.io/Arraymancer/).

## Features

For now Arraymancer is mostly at the ndarray stage, a [vision package](https://github.com/edubart/arraymancer-vision) and a [deep learning demo](https://github.com/edubart/arraymancer-demos) are available with logistic regression and perceptron from scratch.

You can also check the [detailed example](https://github.com/mratsim/Arraymancer/blob/master/examples/ex01_xor_perceptron_from_scratch.nim) or [benchmark](https://github.com/mratsim/Arraymancer/blob/master/benchmarks/ex01_xor.nim) perceptron for a preview of Arraymancer deep learning usage.

Available autograd and neural networks features are detailed in the technical reference part of the [documentation](https://mratsim.github.io/Arraymancer/).

### Arraymancer as a Deep Learning library

Note: The final interface is still **work in progress.**

#### Handwritten digit recognition with Arraymancer
From [example 2](https://github.com/mratsim/Arraymancer/blob/master/examples/ex02_handwritten_digits_recognition.nim).

```Nim
import arraymancer, random

randomize(42) # Random seed for reproducibility

let
  ctx = newContext Tensor[float32] # Autograd/neural network graph
  n = 32                           # Batch size

let
  mnist = load_mnist()
  x_train = mnist.train_images.astype(float32) / 255'f32
  X_train = ctx.variable x_train.unsqueeze(1) # Change shape from [N, H, W] to [N, C, H, W], with C = 1

  y_train = mnist.train_labels.astype(int)

  x_test = mnist.test_images.astype(float32) / 255'f32
  X_test = ctx.variable x_test.unsqueeze(1) Change shape from [N, H, W] to [N, C, H, W], with C = 1
  y_test = mnist.test_labels.astype(int)

network ctx, DemoNet:
  layers:
    x:          Input([1, 28, 28])
    cv1:        Conv2D(x.out_shape, 20, 5, 5)
    mp1:        MaxPool2D(cv1.out_shape, (2,2), (0,0), (2,2))
    cv2:        Conv2D(mp1.out_shape, 50, 5, 5)
    mp2:        MaxPool2D(cv2.out_shape, (2,2), (0,0), (2,2))
    fl:         Flatten(mp2.out_shape)
    hidden:     Linear(fl.out_shape, 500)
    classifier: Linear(500, 10)
  forward x:
    x.cv1.relu.mp1.cv2.relu.mp2.fl.hidden.relu.classifier

let model = ctx.init(DemoNet)
let optim = model.optimizerSGD(learning_rate = 0.01'f32)

# Learning loop
for epoch in 0 ..< 5:
  for batch_id in 0 ..< X_train.value.shape[0] div n: # some at the end may be missing, oh well ...
    # minibatch offset in the Tensor
    let offset = batch_id * n
    let x = X_train[offset ..< offset + n, _]
    let target = y_train[offset ..< offset + n]

    let clf = model.forward(x)
    let loss = clf.sparse_softmax_cross_entropy(target)

    if batch_id mod 200 == 0:
      # Print status every 200 batches
      echo "Epoch is: " & $epoch
      echo "Batch id: " & $batch_id
      echo "Loss is:  " & $loss.value.data[0]

    loss.backprop()
    optim.update()

  # Validation (checking the accuracy/generalization of our model on unseen data)
  ctx.no_grad_mode:
    echo "\nEpoch #" & $epoch & " done. Testing accuracy"

    # Validation by batches of 1000 images
    var score = 0.0
    var loss = 0.0
    for i in 0 ..< 10:
      let y_pred = model.forward(X_test[i*1000 ..< (i+1)*1000, _]).value.softmax.argmax(axis = 1).squeeze
      score += accuracy_score(y_test[i*1000 ..< (i+1)*1000], y_pred)

      loss += model.forward(X_test[i*1000 ..< (i+1)*1000, _]).sparse_softmax_cross_entropy(y_test[i*1000 ..< (i+1)*1000]).value.data[0]
    score /= 10
    loss /= 10
    echo "Accuracy: " & $(score * 100) & "%"
    echo "Loss:     " & $loss
    echo "\n"



############# Output ############

Epoch is: 0
# Batch id: 0
# Loss is:  194.3991851806641
# Epoch is: 0
# Batch id: 200
# Loss is:  2.60599946975708
# Epoch is: 0
# Batch id: 400
# Loss is:  1.708131313323975
# Epoch is: 0
# Batch id: 600
# Loss is:  1.061241149902344
# Epoch is: 0
# Batch id: 800
# Loss is:  0.8607467412948608
# Epoch is: 0
# Batch id: 1000
# Loss is:  0.9292868375778198
# Epoch is: 0
# Batch id: 1200
# Loss is:  0.6178927421569824
# Epoch is: 0
# Batch id: 1400
# Loss is:  0.4008050560951233
# Epoch is: 0
# Batch id: 1600
# Loss is:  0.2450754344463348
# Epoch is: 0
# Batch id: 1800
# Loss is:  0.3787734508514404

# Epoch #0 done. Testing accuracy
# Accuracy: 84.24999999999999%
# Loss:     0.4853884726762772


# Epoch is: 1
# Batch id: 0
# Loss is:  0.8319419622421265
# Epoch is: 1
# Batch id: 200
# Loss is:  0.3116425573825836
# Epoch is: 1
# Batch id: 400
# Loss is:  0.232885867357254
# Epoch is: 1
# Batch id: 600
# Loss is:  0.3877259492874146
# Epoch is: 1
# Batch id: 800
# Loss is:  0.3621436357498169
# Epoch is: 1
# Batch id: 1000
# Loss is:  0.5054937601089478
# Epoch is: 1
# Batch id: 1200
# Loss is:  0.4431287050247192
# Epoch is: 1
# Batch id: 1400
# Loss is:  0.2153264284133911
# Epoch is: 1
# Batch id: 1600
# Loss is:  0.1401071697473526
# Epoch is: 1
# Batch id: 1800
# Loss is:  0.3415909707546234

# Epoch #1 done. Testing accuracy
# Accuracy: 87.91%
# Loss:     0.3657706841826439
```


### Tensors on CPU, on Cuda and OpenCL
Tensors, CudaTensors and CLTensors do not have the same features implemented yet.
Also CudaTensors and CLTensors can only be float32 or float64 while Cpu Tensor can be integers, string, boolean or any custom object.

Here is a comparative table of the core features, not that this feature set is developing
rapidly.

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

Advanced features built upon this are:
  - Neural networks: Dense and Convolutional neural networks are supported on CPU. Primitives are available on Cuda.
  - Linear algebra: Least squares solver and eigenvalue decomposition for symmetric matrices.
  - Machine Learning: Accuracy score, common loss function (MAE, MSE, ...), Principal Component Analysis (PCA).
  - Statistics: Covariance matrix.
  - IO & Datasets: CSV reading and writing, and reading MNIST files.
  - A tensor plotting tool using Python matplotlib.

Detailed API is available at Arraymancer [documentation](https://mratsim.github.io/Arraymancer/).

### Speed

Arraymancer is fast, how it achieves its speed under the hood is detailed [here](https://mratsim.github.io/Arraymancer/uth.speed.html).

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
Arraymancer 0.2.90 (master branch 2017-10-31)

no OpenMP compilation: nim c -d:native -d:release --out:build/integer_matmul --nimcache:./nimcache benchmarks/integer_matmul.nim
with OpenMP: nim c -d:openmp --cc:gcc --gcc.exe:"/usr/local/bin/gcc-6" --gcc.linkerexe:"/usr/local/bin/gcc-6"  -d:native -d:release --out:build/integer_matmul --nimcache:./nimcache benchmarks/integer_matmul.nim
```

| Language | Speed | Memory |
|---|---|---|
| Nim 0.18.0 (devel) - GCC 6 + OpenMP | **0.95s** | 71.9 MB |
| Nim 0.18.0 (devel) - Apple Clang 9 - no OpenMP | **1.73s** | 71.7 MB |
| Julia v0.6.0 | 4.49s | 185.2 MB |
| Python 3.5.2 + Numpy 1.12 | 9.49s | 55.8 MB |

Benchmark setup is in the `./benchmarks` folder and similar to (stolen from) [Kostya's](https://github.com/kostya/benchmarks#matmul). Note: Arraymancer float matmul is as fast as `Julia Native Thread`.

#### Logistic regression
On the [demo benchmark](https://github.com/edubart/arraymancer-demos), Arraymancer is faster than Torch in v0.2.90.

CPU

| Framework | Backend | Forward+Backward Pass Time  |
|---|---|---|
| Arraymancer v0.2.90| OpenMP + MKL | **0.458ms**  |
| Torch7 | MKL | 0.686ms  |
| Numpy | MKL | 0.723ms  |

GPU

| Framework | Backend | Forward+Backward Pass Time  |
|---|---|---|
| Arraymancer v0.2.90| Cuda | WIP  |
| Torch7 | Cuda | 0.286ms  |

#### DNN - 3 hidden layers

CPU

| Framework | Backend | Forward+Backward Pass Time  |
|---|---|---|
| Arraymancer v0.2.90| OpenMP + MKL | **2.907ms**  |
| PyTorch | MKL | 6.797ms  |

GPU

| Framework | Backend | Forward+Backward Pass Time  |
|---|---|---|
| Arraymancer v0.2.90| Cuda | WIP |
| PyTorch | Cuda | 4.765ms  |


```
Intel(R) Core(TM) i7-3770K CPU @ 3.50GHz, gcc 7.2.0, MKL 2017.17.0.4.4, OpenBLAS 0.2.20, Cuda 8.0.61, Geforce GTX 1080 Ti, Nim 0.18.0
```

In the future, Arraymancer will leverage Nim compiler to automatically fuse operations
like `alpha A*B + beta C` or a combination of element-wise operations. This is already done to fuse `toTensor` and `reshape`.
