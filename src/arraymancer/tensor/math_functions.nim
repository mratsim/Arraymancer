# Copyright 2017 the Arraymancer contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import  ./data_structure,
        ./backend/openmp,
        ./init_cpu,
        ./higher_order_applymap,
        ./ufunc
import complex except Complex64, Complex32

# Non-operator math functions

proc elwise_mul*[T](a, b: Tensor[T]): Tensor[T] {.noinit.} =
  ## Element-wise multiply
  map2_inline(a, b, x * y)

proc melwise_mul*[T](a: var Tensor[T], b: Tensor[T]) =
  ## Element-wise multiply
  a.apply2_inline(b, x * y)

proc elwise_div*[T: SomeInteger](a, b: Tensor[T]): Tensor[T] {.noinit.} =
  ## Element-wise division
  map2_inline(a, b, x div y)

proc elwise_div*[T: SomeFloat](a, b: Tensor[T]): Tensor[T] {.noinit.} =
  ## Element-wise division
  map2_inline(a, b, x / y)

proc melwise_div*[T: SomeInteger](a: var Tensor[T], b: Tensor[T]) =
  ## Element-wise division (in-place)
  a.apply2_inline(b, x div y)

proc melwise_div*[T: SomeFloat](a: var Tensor[T], b: Tensor[T]) =
  ## Element-wise division (in-place)
  a.apply2_inline(b, x / y)

proc reciprocal*[T: SomeFloat](t: Tensor[T]): Tensor[T] {.noinit.} =
  ## Return a tensor with the reciprocal 1/x of all elements
  t.map_inline(1.T/x)

proc mreciprocal*[T: SomeFloat](t: var Tensor[T]) =
  ## Apply the reciprocal 1/x in-place to all elements of the Tensor
  t.apply_inline(1.T/x)

proc reciprocal*[T: Complex[float32] or Complex[float64]](t: Tensor[T]): Tensor[T] {.noinit.} =
  ## Return a tensor with the reciprocal 1/x of all elements
  type F = T.T # Get float subtype of Complex[T]
  t.map_inline(complex(1.F, 0.F)/x)

proc mreciprocal*[T: Complex[float32] or Complex[float64]](t: var Tensor[T]) =
  ## Apply the reciprocal 1/x in-place to all elements of the Tensor
  type F = T.T # Get float subtype of Complex[T]
  t.apply_inline(complex(1.F, 0.F)/x)

proc negate*[T: SomeSignedInt|SomeFloat](t: Tensor[T]): Tensor[T] {.noinit.} =
  ## Return a tensor with all elements negated (10 -> -10)
  t.map_inline(-x)

proc mnegate*[T: SomeSignedInt|SomeFloat](t: var Tensor[T]) =
  ## Negate in-place all elements of the tensor (10 -> -10)
  t.apply_inline(-x)

proc `-`*[T: SomeNumber](t: Tensor[T]): Tensor[T] {.noinit.} =
  ## Negate all values of a Tensor
  t.map_inline(-x)

# Built-in nim function that doesn't work with makeUniversal
proc abs*[T:SomeNumber](t: Tensor[T]): Tensor[T] {.noinit.} =
  ## Return a Tensor with absolute values of all elements
  t.map_inline(abs(x))

# complex abs -> float
proc abs*(t: Tensor[Complex[float64]]): Tensor[float64] {.noinit.} =
  ## Return a Tensor with absolute values of all elements
  t.map_inline(abs(x))

proc abs*(t: Tensor[Complex[float32]]): Tensor[float32] {.noinit.} =
  ## Return a Tensor with absolute values of all elements
  t.map_inline(abs(x))

proc mabs*[T](t: var Tensor[T]) =
  ## Return a Tensor with absolute values of all elements
  # FIXME: how to inplace convert Tensor[Complex] to Tensor[float]
  t.apply_inline(abs(x))

# complex phase -> float
proc phase*(t: Tensor[Complex[float64]]): Tensor[float64] {.noinit.} =
  ## Return a Tensor with phase values of all elements
  t.map_inline(phase(x))

proc phase*(t: Tensor[Complex[float32]]): Tensor[float32] {.noinit.} =
  ## Return a Tensor with phase values of all elements
  t.map_inline(phase(x))

proc clamp*[T](t: Tensor[T], min, max: T): Tensor[T] {.noinit.} =
  ## Return a Tensor with all elements clamped to the interval [min, max].
  t.map_inline(clamp(x, min, max))

proc mclamp*[T](t: var Tensor[T], min, max: T) =
  ## Update the Tensor with all elements clamped to the interval [min, max].
  t.apply_inline(clamp(x, min, max))

proc square*[T](x: T): T {.inline.} =
  ## Return `x*x`
  x*x

makeUniversal(square)

type ConvolveMode* = enum full, same, valid

proc convolveImpl[T: SomeNumber | Complex32 | Complex64](
    f, g: Tensor[T],
    mode: ConvolveMode): Tensor[T] {.noinit.} =
  ## Implementation of the linear convolution of two one-dimensional tensors

  # Calculate the result lenth and the shift offset
  let len_result = case mode
    of full: f.size + g.size - 1
    of same: max(f.size, g.size)
    of valid: max(f.size, g.size) - min(f.size, g.size) + 1
  let offset = case mode
    of full: 0
    of same: (min(f.size, g.size) - 1) div 2
    of valid: min(f.size, g.size) - 1

  # Initialize the result tensor
  result = zeros[T](len_result)

  # And perform the convolution
  omp_parallel_blocks(block_offset, block_size, len_result):
    for n in block_offset ..< block_offset + block_size:
      let shift = n + offset
      for m in max(0, shift - g.size + 1) .. min(f.size - 1, shift):
        result[n] += f[m] * g[shift - m]

proc convolve*[T: SomeNumber | Complex32 | Complex64](
    t1, t2: Tensor[T],
    mode = ConvolveMode.full): Tensor[T] {.noinit.} =
  ## Returns the discrete, linear convolution of two one-dimensional tensors.
  ##
  ## The convolution operator is often seen in signal processing, where it models
  ## the effect of a linear time-invariant system on a signal (Wikipedia,
  ## “Convolution”, https://en.wikipedia.org/wiki/Convolution).
  ##
  ## The convolution is defined as the integral of the product of the two tensors
  ## after one is reflected about the y-axis and shifted n positions, for all values
  ## of n in which the tensors overlap (since the integral will be zero outside of
  ## that window).
  ##
  ## Inputs:
  ##   - t1, t2: Input tensors of size N and M respectively.
  ##   - mode: Convolution mode (full, same, valid):
  ##     - `full`: This is the default mode. It returns the convolution at each point
  ##               of overlap, with an output shape of (N+M-1,). At the end-points of
  ##               the convolution, the signals do not overlap completely, and boundary
  ##               effects may be seen.
  ##      - `same`: Returns an output of length max(M, N).
  ##                Boundary effects are still visible.
  ##      - `valid`: Returns output of length max(M, N) - min(M, N) + 1.
  ##                 The convolution is only given for points where the signals overlap
  ##                 completely. Values outside the signal boundary have no effect.
  ##
  ## Output:
  ##   - Convolution tensor of same type as the inputs and size according to the mode.
  ##
  ## Notes:
  ##  - The API of this function is the same as the one of numpy.convolve.

  # Ensure that both arrays are 1-dimensional
  let f = if t1.rank > 1: t1.squeeze else: t1
  let g = if t2.rank > 1: t2.squeeze else: t2
  if f.rank > 1:
    raise newException(ValueError,
      "convolve input tensors must be 1D, but first input tensor is multi-dimensional (shape=" & $t1.shape & ")")
  if g.rank > 1:
    raise newException(ValueError,
      "convolve input tensors must be 1D, but second input tensor is multi-dimensional (shape=" & $t2.shape & ")")

  convolveImpl(f, g, mode=mode)

type CorrelateMode* = ConvolveMode

proc correlate*[T: SomeNumber](
    t1, t2: Tensor[T],
    mode = CorrelateMode.valid): Tensor[T] {.noinit.} =
  ## Returns the cross-correlation of two one-dimensional real tensors.
  ##
  ## The correlation is defined as the integral of the product of the two tensors
  ## after the second one is shifted n positions, for all values of n in which
  ## the tensors overlap (since the integral will be zero outside of that window).
  ##
  ## Inputs:
  ##   - t1, t2: Input tensors of size N and M respectively.
  ##   - mode: Correlation mode (full, same, valid):
  ##     - `full`: It returns the correlation at each point
  ##               of overlap, with an output shape of (N+M-1,). At the end-points of
  ##               the correlation, the signals do not overlap completely, and boundary
  ##               effects may be seen.
  ##      - `same`: Returns an output of length max(M, N).
  ##                Boundary effects are still visible.
  ##      - `valid`: This is the default mode. Returns output of length max(M, N) - min(M, N) + 1.
  ##                 The correlation is only given for points where the signals overlap
  ##                 completely. Values outside the signal boundary have no effect.
  ##
  ## Output:
  ##   - Correlation tensor of same type as the inputs and size according to the mode.
  ##
  ## Notes:
  ##   - The API of this function is the same as the one of numpy.correlate.
  ##   - Note that (as with np.correlate) the default correlation mode is `valid`,
  ##     which is different than the default convolution mode (`full`).

  # Ensure that both arrays are 1-dimensional
  let f = if t1.rank > 1: t1.squeeze else: t1
  let g = if t2.rank > 1: t2.squeeze else: t2
  if f.rank > 1:
    raise newException(ValueError,
      "correlate input tensors must be 1D, but first input tensor is multi-dimensional (shape=" & $t1.shape & ")")
  if g.rank > 1:
    raise newException(ValueError,
      "correlate input tensors must be 1D, but second input tensor is multi-dimensional (shape=" & $t2.shape & ")")
  mixin `|-`
  mixin `_`
  convolveImpl(f, g[_|-1], mode=mode)

proc correlate*[T: Complex32 | Complex64](
    t1, t2: Tensor[T],
    mode = CorrelateMode.valid): Tensor[T] {.noinit.} =
  ## Returns the cross-correlation of two one-dimensional complex tensors.
  ##
  ## The correlation is defined as the integral of the product of the two tensors
  ## after the second one is conjugated and shifted n positions, for all values
  ## of n in which the tensors overlap (since the integral will be zero outside of
  ## that window).
  ##
  ## Inputs:
  ##   - t1, t2: Input tensors of size N and M respectively.
  ##   - mode: Correlation mode (full, same, valid):
  ##     - `full`: It returns the correlation at each point
  ##               of overlap, with an output shape of (N+M-1,). At the end-points of
  ##               the correlation, the signals do not overlap completely, and boundary
  ##               effects may be seen.
  ##      - `same`: Returns an output of length max(M, N).
  ##                Boundary effects are still visible.
  ##      - `valid`: This is the default mode. Returns output of length max(M, N) - min(M, N) + 1.
  ##                 The correlation is only given for points where the signals overlap
  ##                 completely. Values outside the signal boundary have no effect.
  ##
  ## Output:
  ##   - Correlation tensor of same type as the inputs and size according to the mode.
  ##
  ## Notes:
  ##   - The API of this function is the same as the one of numpy.correlate.
  ##   - Note that (as with np.correlate) the default correlation mode is `valid`,
  ##     which is different than the default convolution mode (`full`).

  # Ensure that both arrays are 1-dimensional
  let f = if t1.rank > 1: t1.squeeze else: t1
  let g = if t2.rank > 1: t2.squeeze else: t2
  if f.rank > 1:
    raise newException(ValueError,
      "correlate input tensors must be 1D, but first input tensor is multi-dimensional (shape=" & $t1.shape & ")")
  if g.rank > 1:
    raise newException(ValueError,
      "correlate input tensors must be 1D, but second input tensor is multi-dimensional (shape=" & $t2.shape & ")")
  mixin `|-`
  mixin `_`
  convolveImpl(f, g[_|-1].conjugate, mode=mode)
