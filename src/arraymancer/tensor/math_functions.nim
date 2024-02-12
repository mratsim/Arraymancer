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
import math

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

proc sgn*[T: SomeNumber](t: Tensor[T]): Tensor[int] {.noinit.} =
  ## Element-wise sgn function (returns a tensor with the sign of each element)
  ##
  ## Returns:
  ## - -1 for negative numbers and NegInf,
  ## - 1 for positive numbers and Inf,
  ## - 0 for positive zero, negative zero and NaN
  t.map_inline(sgn(x))

when (NimMajor, NimMinor, NimPatch) >= (1, 6, 0):
  proc copySign*[T: SomeFloat](t1, t2: Tensor[T]): Tensor[T] {.noinit.} =
    ## Element-wise copySign function (combines 2 tensors, taking the magnitudes from t1 and the signs from t2)
    ##
    ## This uses nim's copySign under the hood, and thus has the same properties. That is, it works for values
    ## which are NaN, infinity or zero (all of which can carry a sign) but does not work for integers.
    t1.map2_inline(t2, copySign(x, y))

  proc mcopySign*[T: SomeFloat](t1: var Tensor[T], t2: Tensor[T]) =
    ## In-place element-wise copySign function (changes the signs of the elements of t1 to match those of t2)
    ##
    ## This uses nim's copySign under the hood, and thus has the same properties. That is, it works for values
    ## which are NaN, infinity or zero (all of which can carry a sign) but does not work for integers.
    t1.apply2_inline(t2, copySign(x, y))

proc floorMod*[T: SomeNumber](t1, t2: Tensor[T]): Tensor[T] {.noinit.} =
  ## Broadcasted floorMod operation: floorMod(tensor, tensor).
  result = t1.map2_inline(t2, floorMod(x, y))

proc floorMod*[T: SomeNumber](t: Tensor[T], val: T): Tensor[T] {.noinit.} =
  ## Broadcasted floorMod operation: floorMod(tensor, scalar).
  result = t.map_inline(floorMod(x, val))

proc floorMod*[T: SomeNumber](val: T, t: Tensor[T]): Tensor[T] {.noinit.} =
  ## Broadcasted floorMod operation: floorMod(scalar, tensor).
  result = t.map_inline(floorMod(val, x))

proc clamp*[T](t: Tensor[T], min, max: T): Tensor[T] {.noinit.} =
  ## Return a Tensor with all elements clamped to the interval [min, max].
  t.map_inline(clamp(x, min, max))

proc mclamp*[T](t: var Tensor[T], min, max: T) =
  ## Update the Tensor with all elements clamped to the interval [min, max].
  t.apply_inline(clamp(x, min, max))

proc max*[T: SomeNumber](t1, t2: Tensor[T]): Tensor[T] {.noinit.} =
  ## Compare two arrays and return a new array containing the element-wise maxima.
  ##
  ## As in nim's built-in max procedure if one of the elements being compared is a NaN,
  ## then the non NaN element is returned.
  t1.map2_inline(t2, max(x, y))

proc max*[T: SomeNumber](args: varargs[Tensor[T]]): Tensor[T] {.noinit.} =
  ## Compare any number of arrays and return a new array containing the element-wise maxima.
  ##
  ## As in nim's built-in max procedure if one of the elements being compared is a NaN,
  ## then the non NaN element is returned.
  result = max(args[0], args[1])
  for n in countup(2, len(args)-1):
    result = max(result, args[n])

proc mmax*[T: SomeNumber](t1: var Tensor[T], t2: Tensor[T]) =
  ## In-place element-wise maxima of two tensors.
  ##
  ## As in nim's built-in max procedure if one of the elements being compared is a NaN,
  ## then the non NaN element is returned.
  t1.apply2_inline(t2, max(x, y))

proc mmax*[T: SomeNumber](t1: var Tensor[T], args: varargs[Tensor[T]]) =
  ## In-place element-wise maxima of N tensors.
  ##
  ## As in nim's built-in max procedure if one of the elements being compared is a NaN,
  ## then the non NaN element is returned.
  t1.apply2_inline(args[0], max(x, y))
  for n in countup(1, len(args)-1):
    t1.apply2_inline(args[n], max(x, y))

proc min*[T: SomeNumber](t1, t2: Tensor[T]): Tensor[T] {.noinit.} =
  ## Compare two arrays and return a new array containing the element-wise minima.
  ##
  ## As in nim's built-in min procedure if one of the elements being compared is a NaN,
  ## then the non NaN element is returned.
  t1.map2_inline(t2, min(x, y))

proc min*[T: SomeNumber](args: varargs[Tensor[T]]): Tensor[T] {.noinit.} =
  ## Compare any number of arrays and return a new array containing the element-wise minima.
  ##
  ## As in nim's built-in min procedure if one of the elements being compared is a NaN,
  ## then the non NaN element is returned.
  result = min(args[0], args[1])
  for n in countup(2, len(args)-1):
    result = min(result, args[n])

proc mmin*[T: SomeNumber](t1: var Tensor[T], t2: Tensor[T]) =
  ## In-place element-wise minima of two tensors.
  ##
  ## As in nim's built-in min procedure if one of the elements being compared is a NaN,
  ## then the non NaN element is returned.
  t1.apply2_inline(t2, min(x, y))

proc mmin*[T: SomeNumber](t1: var Tensor[T], args: varargs[Tensor[T]]) =
  ## In-place element-wise minima of N tensors.
  ##
  ## As in nim's built-in min procedure if one of the elements being compared is a NaN,
  ## then the non NaN element is returned.
  t1.apply2_inline(args[0], min(x, y))
  for n in countup(1, len(args)-1):
    t1.apply2_inline(args[n], min(x, y))

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
