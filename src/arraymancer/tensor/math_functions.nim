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
        ./ufunc,
        ./operators_blas_l2l3
import std / math
import complex except Complex64, Complex32

export math

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

proc sinc*[T: SomeFloat](x: T, normalized: static bool = true): T {.inline.} =
  ## Return the normalized or non-normalized sinc function.
  ##
  ## For values other than 0, the normalized sinc function is equal to
  ## `sin(PI * x) / (PI * x)`, while the non-normalized sync function
  ## is equal to `sin(x) / x`.
  ## `sinc(0)` takes the limit value 1 in both cases, making `sinc` not only
  ## everywhere continuous but also infinitely differentiable.
  ##
  ## Inputs:
  ## - t: Real input value.
  ## - normalized: Select whether to return the normalized or non-normalized sync.
  ##               This argument is static so it must be set at compile time.
  ##               The default is `true` (i.e. to return the normalized sync).
  ## Result:
  ## - Calculated sinc value
  if x == 0.T:
    return 1.T
  when normalized:
    let x = T(PI) * x
  sin(x) / x

proc sinc*[T: SomeFloat](t: Tensor[T], normalized: static bool = true): Tensor[T] {.noinit.} =
  ## Return the normalized or non-normalized sinc function of a Tensor
  ##
  ## For values other than 0, the normalized sinc function is equal to
  ## `sin(PI * x) / (PI * x)`, while the non-normalized sync function
  ## is equal to `sin(x) / x`.
  ## `sinc(0)` takes the limit value 1 in both cases, making `sinc` not only
  ## everywhere continuous but also infinitely differentiable.
  ##
  ## Inputs:
  ## - t: Input real tensor.
  ## - normalized: Select whether to return the normalized or non-normalized sync.
  ##               This argument is static so it must be set at compile time.
  ##               The default is `true` (i.e. to return the normalized sync).
  ## Result:
  ## - New tensor with the sinc values of all the input tensor elements.
  t.map_inline(sinc(x, normalized=normalized))

proc classify*[T: SomeFloat](t: Tensor[T]): Tensor[FloatClass] {.noinit.} =
  ## Element-wise classify function (returns a tensor with the float class of each element).
  ##
  ## Returns:
  ##   A FloatClass tensor where each value is one of the following:
  ##   - fcNormal: value is an ordinary nonzero floating point value
  ##   - fcSubnormal: value is a subnormal (a very small) floating point value
  ##   - fcZero: value is zero
  ##   - fcNegZero: value is the negative zero
  ##   - fcNan: value is Not a Number (NaN)
  ##   - fcInf: value is positive infinity
  ##   - fcNegInf: value is negative infinity
  t.map_inline(classify(x))

proc almostEqual*[T: SomeFloat | Complex32 | Complex64](t1, t2: Tensor[T],
    unitsInLastPlace: Natural = 4): Tensor[bool] {.noinit.} =
  ## Element-wise almostEqual function
  ##
  ## Checks whether pairs of elements of two tensors are almost equal, using
  ## the [machine epsilon](https://en.wikipedia.org/wiki/Machine_epsilon).
  ##
  ## For more details check the section covering the `almostEqual` procedure in
  ## nim's standard library documentation.
  ##
  ## Inputs:
  ## - t1, t2: Input (floating point or complex) tensors of the same shape.
  ## - unitsInLastPlace: The max number of
  ##                     [units in the last place](https://en.wikipedia.org/wiki/Unit_in_the_last_place)
  ##                     difference tolerated when comparing two numbers. The
  ##                     larger the value, the more error is allowed. A `0`
  ##                     value means that two numbers must be exactly the
  ##                     same to be considered equal.
  ##
  ## Result:
  ## - A new boolean tensor of the same shape as the inputs, in which elements
  ##   are true if the two values in the same position on the two input tensors
  ##   are almost equal (and false if they are not).
  ##
  ## Note:
  ## - You can combine this function with `all` to check if two real tensors
  ##   are almost equal.
  map2_inline(t1, t2):
    when T is Complex:
      almostEqual(x.re, y.re, unitsInLastPlace=unitsInLastPlace) and
      almostEqual(x.im, y.im, unitsInLastPlace=unitsInLastPlace)
    else:
      almostEqual(x, y, unitsInLastPlace=unitsInLastPlace)

# Convolution / Correlation related procedures

proc array2shiftColMatrix[T](input: Tensor[T], kernel_size: int,
                  padding = 0, stride = 1,
                  result: var Tensor[T])  =
  ## Rank-1 (i.e. array) version of conv/im2col
  ##
  ## This function is a version of im2col that only works with rank-1 tensors.
  ## It is about 30% faster than the generic `im2col` version.
  ##
  ## This function takes a rank-1 tensor and generates a "shift column matrix",
  ## which is a matrix in which every column is a "shifted" copy of the input
  ## tensor (with the shift amount increasing by `stride` on every subsequent
  ## column). The amount of shifts depends on the `stride` as well as on the
  ## `padding` which is the total number of zeros that are added around the
  ## input tensor to generate each shift.
  ##
  ## The reason this is done is to be able to perform a convolution by
  ## multiplying this "shift column matrix" by the kernel tensor.
  let
    width = input.len
    num_shifts = (width + (2 * padding) - kernel_size) div stride + 1

  assert result.is_C_contiguous and input.is_C_contiguous
  assert result.shape == [kernel_size, num_shifts]

  let odata = result.unsafe_raw_offset()
  let idata = input.unsafe_raw_offset()
  for c in `||`(0, kernel_size-1, "simd"):
    let
      w_offset = (c mod kernel_size) - padding
      c_offset = c div kernel_size
    for w in 0 ..< num_shifts:
      let col = w_offset + (w * stride)
      when T is Complex64:
        var v = complex64(0.0)
      elif T is Complex32:
        var v = complex32(0.0)
      else:
        var v = 0.T
      if col >= 0 and col < width:
        let iidx = (c_offset * width) + col
        v = idata[iidx]
      let oidx = (c * num_shifts) + w
      odata[oidx] = v

type ConvolveMode* = enum full, same, valid

proc correlateImpl[T](f, g: Tensor[T],
                      mode: ConvolveMode,
                      stride = 1): Tensor[T] =
  ## Compute the cross-correlation using BLAS' GEMM function
  # Implementation with ideas from http://cs231n.github.io/convolutional-networks/#conv
  if f.size < g.size:
    # Call correlateImpl with both inputs swapped and reversed
    # The reason is as follows:
    # The GEMM based implementation assumes that the first input tensor is not
    # shorter than the second. If it is we must swap the inputs.
    # However, this causes the result of the GEMM operation to be reversed.
    # To avoid this result reversal we can simply reverse the inputs (in
    # addition to swapping them).
    # It would seem that an alternative to reversing both inputs would be to
    # just reverse the result. However this does not work well when using the
    # `same` and `valid` modes or when setting the `down` argument to something
    # other than 1, because in some cases the output is then shifted by 1 sample
    mixin `|-`
    mixin `_`
    result = correlateImpl(g[_|-1], f[_|-1], mode = mode, stride = stride)
    return result

  let f = f.asContiguous()
  let g = g.asContiguous()

  # Note that here we know that f is longer or as long as g, therefore we know
  # that `max(f.len, g.len) = f.len` and `min(f.len, g.len) = g.len`!
  let target_result_len = case mode:
    of full: f.len + g.len - 1
    of same: f.len               # i.e. max(f.len, g.len)
    of valid: f.len - g.len + 1  # i.e. max(f.len, g.len) - min(f.len, g.len) + 1

  let padding = case mode:
    of full: g.len - 1  # i.e. min(f.len, g.len) - 1
    of same: ceil((g.len - 1).float / 2.0).int  # i.e. ceil((min(f.len, g.len).float - 1.0) / 2.0).int
    of valid: 0

  let
    result_len = (f.len + (2*padding) - g.len) div stride + 1
    kernel_col = g.reshape(1, g.len)

  # Prepare the `result` tensor whose shape must be `[1, N]`,
  # otherwise the `gemm` call below doesn't do the right thing!
  result = newTensorUninit[T](1, result_len)

  # # Let's make sure both inputs are contiguous
  # let f = f.asContiguous()
  # let g = g.asContiguous()

  # Create the "shifted column input matrix" that will be used to calculate the
  # convolution through a matrix multiplication with the kernel
  var input_shifts = newTensorUninit[T](g.len, result_len)

  array2shiftColMatrix(f, g.len, padding, stride, input_shifts)

  # Perform the actual convolution
  # The following must be done without copy: GEMM will directly write in the result tensor
  when T is Complex64:
    const one = complex64(1.0)
    const zero = complex64(0.0)
  elif T is Complex32:
    const one = complex32(1.0)
    const zero = complex32(0.0)
  else:
    const one = 1.T
    const zero = 0.T
  mixin `_`
  gemm(one, kernel_col, input_shifts, zero, result)

  # Now remove the unnecessary dimension of the result
  result = result.squeeze()

  # And remove the extra samples that sometimes are added because `array2shiftColMatrix`
  # works with symmetric paddings at the start and end of input_shifts
  if target_result_len < result_len:
    result = result[_ ..< target_result_len]

proc convolve*[T](
    t1, t2: Tensor[T],
    mode = ConvolveMode.full,
    down = 1): Tensor[T] {.noinit.} =
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
  ##   - t1, t2: Rank-1 input tensors of size N and M respectively.
  ##   - mode: Convolution mode (`full`, `same` or `valid`):
  ##     - `full`: This is the default mode. It returns the convolution at
  ##               each point of overlap, with an output length of `N + M - 1`.
  ##               At the end-points of the convolution, the signals do not
  ##               overlap completely, and boundary effects may be seen.
  ##      - `same`: Returns an output of length `max(M, N)`.
  ##                Boundary effects are still visible.
  ##      - `valid`: Returns output of length `max(M, N) - min(M, N) + 1`.
  ##                 The convolution is only given for points where the signals
  ##                 overlap completely. Values outside the signal boundary
  ##                 have no effect.
  ##   - down: Downsample ratio applied to the result. Defaults to 1 (i.e.
  ##           no downsampling).
  ##
  ## Result:
  ##   - Convolution tensor of the same type as the inputs and size according
  ##     to the mode and the selected `down` downsample ratio.
  ##
  ## Notes:
  ##   - The API of this function is based on `numpy.convolve`, with the
  ##     addtion of the `down` argument.
  ##   - The `down` argument is useful for certain signal processing tasks,
  ##     and is more efficient than applying the downsampling after the
  ##     convolution step (which requires `down` times more operations).

  # Ensure that both arrays are 1-dimensional
  let f = if t1.rank > 1: t1.squeeze else: t1
  let g = if t2.rank > 1: t2.squeeze else: t2
  if f.rank > 1:
    raise newException(ValueError,
      "convolve input tensors must be 1D, but first input tensor is multi-dimensional (shape=" & $t1.shape & ")")
  if g.rank > 1:
    raise newException(ValueError,
      "convolve input tensors must be 1D, but second input tensor is multi-dimensional (shape=" & $t2.shape & ")")
  mixin `|-`
  mixin `_`
  correlateImpl(f, g[_|-1], mode = mode, stride = down)

type CorrelateMode* = ConvolveMode

proc correlate*[T: SomeNumber | Complex32 | Complex64](
    t1, t2: Tensor[T],
    mode = CorrelateMode.valid,
    down = 1): Tensor[T] {.noinit.} =
  ## Returns the cross-correlation of two one-dimensional real tensors.
  ##
  ## The correlation is defined as the integral of the product of the two
  ## tensors after the second one is shifted n positions, for all values of n
  ## in which the tensors overlap (since the integral will be zero outside of
  ## that window).
  ##
  ## Inputs:
  ##   - t1, t2: Rank-1 input tensors of size N and M respectively.
  ##   - mode: Correlation mode (`full`, `same` or `valid`):
  ##     - `full`: It returns the correlation at each point
  ##               of overlap, with an output length of `N + M - 1`.
  ##               At the end-points of the correlation, the signals do not
  ##                overlap completely, and boundary effects may be seen.
  ##      - `same`: Returns an output of length `max(M, N)`.
  ##                Boundary effects are still visible.
  ##      - `valid`: This is the default mode. Returns output of length
  ##                 `max(M, N) - min(M, N) + 1`.
  ##                 The correlation is only given for points where the signals
  ##                 overlap completely. Values outside the signal boundary
  ##                 have no effect.
  ##   - down: Downsample ratio applied to the result. Defaults to 1 (i.e.
  ##           no downsampling).
  ##
  ## Result:
  ##   - Correlation tensor of the same type as the inputs and size according
  ##     to the mode and the selected `down` downsample ratio.
  ##
  ## Notes:
  ##   - Note that (as with np.correlate) the default correlation mode is
  ##     `valid`, which is different than the default convolution mode (`full`).
  ##   - The API of this function is based on `numpy.convolve`, with the
  ##     addtion of the `down` argument.
  ##   - The `down` argument is useful for certain signal processing tasks,
  ##     and is more efficient than applying the downsampling after the
  ##     correlation step (which requires `down` times more operations).

  # Ensure that both arrays are 1-dimensional
  let f = if t1.rank > 1: t1.squeeze else: t1
  let g = if t2.rank > 1: t2.squeeze else: t2
  if f.rank > 1:
    raise newException(ValueError,
      "correlate input tensors must be 1D, but first input tensor is multi-dimensional (shape=" & $t1.shape & ")")
  if g.rank > 1:
    raise newException(ValueError,
      "correlate input tensors must be 1D, but second input tensor is multi-dimensional (shape=" & $t2.shape & ")")
  when T is Complex:
    correlateImpl(f, g.conjugate, mode=mode, stride = down)
  else:
    correlateImpl(f, g, mode=mode, stride = down)
