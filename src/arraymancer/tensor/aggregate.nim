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
        ./init_cpu,
        ./higher_order_foldreduce,
        ./operators_broadcasted,
        ./operators_comparison,
        ./higher_order_applymap,
        ./math_functions,
        ./accessors,
        ./accessors_macros_syntax,
        ./algorithms,
        ./private/p_empty_tensors
import std/[math, macros]
import complex except Complex64, Complex32

# ### Standard aggregate functions
# TODO consider using stats from Nim standard lib: https://nim-lang.org/docs/stats.html#standardDeviation,RunningStat

# Note: for aggregate that returns scalar, if the tensor is empty,
#       Numpy seems to return the neutral element, do we want that?


proc sum*[T](arg: Tensor[T]): T =
  ## Compute the sum of all elements
  arg.reduce_inline():
    x+=y

proc sum*[T](arg: Tensor[T], axis: int): Tensor[T] {.noinit.} =
  ## Compute the sum of all elements along an axis
  returnEmptyIfEmpty(arg)
  arg.reduce_axis_inline(axis):
    x+=y

proc product*[T](arg: Tensor[T]): T =
  ## Compute the product of all elements
  arg.reduce_inline():
    x*=y

proc product*[T](arg: Tensor[T], axis: int): Tensor[T] {.noinit.}=
  ## Compute the product along an axis
  returnEmptyIfEmpty(arg)
  arg.reduce_axis_inline(axis):
    x.melwise_mul(y)

proc mean*[T: SomeInteger](arg: Tensor[T]): T {.inline.}=
  ## Compute the mean of all elements
  ##
  ## Warning ⚠: Since input is integer, output will also be integer (using integer division)
  arg.sum div arg.size.T

proc mean*[T: SomeInteger](arg: Tensor[T], axis: int): Tensor[T] {.noinit,inline.}=
  ## Compute the mean along an axis
  ##
  ## Warning ⚠: Since input is integer, output will also be integer (using integer division)
  returnEmptyIfEmpty(arg)
  arg.sum(axis) div arg.shape[axis].T

proc mean*[T: SomeFloat](arg: Tensor[T]): T {.inline.}=
  ## Compute the mean of all elements
  arg.sum / arg.size.T

proc mean*[T: Complex[float32] or Complex[float64]](arg: Tensor[T]): T {.inline.}=
  ## Compute the mean of all elements
  type F = T.T # Get float subtype of Complex[T]
  arg.sum / complex(arg.size.F, 0.F)

proc mean*[T: SomeFloat](arg: Tensor[T], axis: int): Tensor[T] {.noinit,inline.}=
  ## Compute the mean along an axis
  returnEmptyIfEmpty(arg)
  arg.sum(axis) / arg.shape[axis].T

proc mean*[T: Complex[float32] or Complex[float64]](arg: Tensor[T], axis: int): Tensor[T] {.noinit,inline.}=
  ## Compute the mean along an axis
  returnEmptyIfEmpty(arg)
  type F = T.T # Get float subtype of Complex[T]
  arg.sum(axis) / complex(arg.shape[axis].F, 0.F)

proc min*[T](arg: Tensor[T]): T =
  ## Compute the min of all elements
  reduce_inline(arg):
    x = min(x,y)

proc min*[T](arg: Tensor[T], axis: int): Tensor[T] {.noinit.} =
  ## Compute the min along an axis
  returnEmptyIfEmpty(arg)
  arg.reduce_axis_inline(axis):
    for ex, ey in mzip(x,y):
      ex = min(ex,ey)

proc max*[T](arg: Tensor[T]): T =
  ## Compute the max of all elements
  arg.reduce_inline():
    x = max(x,y)

proc max*[T](arg: Tensor[T], axis: int): Tensor[T] {.noinit.} =
  ## Compute the max along an axis
  returnEmptyIfEmpty(arg)
  arg.reduce_axis_inline(axis):
    for ex, ey in mzip(x,y):
      ex = max(ex,ey)

proc variance*[T: SomeFloat](arg: Tensor[T]): T =
  ## Compute the sample variance of all elements
  ## The normalization is by (n-1), also known as Bessel's correction,
  ## which partially correct the bias of estimating a population variance from a sample of this population.
  let mean = arg.mean()
  result = arg.fold_inline() do:
    # Initialize to the first element
    x = square(y - mean)
  do:
    # Fold in parallel by summing remaning elements
    x += square(y - mean)
  do:
    # Merge parallel folds
    x += y
  result /= (arg.size-1).T

proc variance*[T: SomeFloat](arg: Tensor[T], axis: int): Tensor[T] {.noinit.} =
  ## Compute the variance of all elements
  ## The normalization is by the (n-1), like in the formal definition
  returnEmptyIfEmpty(arg)
  let mean = arg.mean(axis)
  result = arg.fold_axis_inline(Tensor[T], axis) do:
    # Initialize to the first element
    x = square(y - mean)
  do:
    # Fold in parallel by summing remaning elements
    for ex, ey, em in mzip(x,y,mean):
      ex += square(ey - em)
  do:
    # Merge parallel folds
    x += y
  result /= (arg.shape[axis]-1).T

proc std*[T: SomeFloat](arg: Tensor[T]): T {.inline.} =
  ## Compute the standard deviation of all elements
  ## The normalization is by the (n-1), like in the formal definition
  sqrt(arg.variance())

proc std*[T: SomeFloat](arg: Tensor[T], axis: int): Tensor[T] {.noinit,inline.} =
  ## Compute the standard deviation of all elements
  ## The normalization is by the (n-1), like in the formal definition
  returnEmptyIfEmpty(arg)
  sqrt(arg.variance(axis))

proc argmax_max*[T: SomeNumber](arg: Tensor[T], axis: int): tuple[indices: Tensor[int], maxes: Tensor[T]] {.noinit.} =
  ## Returns (indices, maxes) along an axis
  ##
  ## Input:
  ##   - A tensor
  ##   - An axis (int)
  ##
  ## Returns:
  ##   - A tuple of tensors (indices, maxes) along this axis
  ##
  ## Example:
  ##   .. code:: nim
  ##     let a = [[0, 4, 7],
  ##              [1, 9, 5],
  ##              [3, 4, 1]].toTensor
  ##     assert argmax(a, 0).indices == [[2, 1, 0]].toTensor
  ##     assert argmax(a, 1).indices == [[2],
  ##                                     [1],
  ##                                     [1]].toTensor

  if arg.size == 0:
    result.indices.reset()
    result.maxes.reset()

  result.maxes = arg.atAxisIndex(axis, 0).clone()
  result.indices = zeros[int](result.maxes.shape)

  let dmax = result.maxes.unsafe_raw_buf()
  let dind = result.indices.unsafe_raw_buf()

  for i, subtensor in arg.enumerateAxis(axis, 1, arg.shape[axis] - 1):
    for j, val in enumerate(subtensor):
      if val > dmax[j]:
        dind[j] = i
        dmax[j] = val

proc argmax*[T](arg: Tensor[T], axis: int): Tensor[int] {.inline.}=
  ## Returns the index of the maximum along an axis
  ##
  ## Input:
  ##   - A tensor
  ##   - An axis (int)
  ##
  ## Returns:
  ##   - A tensor of index of the maximums along this axis
  ##
  ## Example:
  ##   .. code:: nim
  ##     let a = [[0, 4, 7],
  ##              [1, 9, 5],
  ##              [3, 4, 1]].toTensor
  ##     assert argmax(a, 0) == [[2, 1, 0]].toTensor
  ##     assert argmax(a, 1) == [[2],
  ##                             [1],
  ##                             [1]].toTensor
  argmax_max(arg, axis).indices

proc argmin_min*[T: SomeNumber](arg: Tensor[T], axis: int): tuple[indices: Tensor[int], mins: Tensor[T]] {.noinit.} =
  ## Returns (indices, mins) along an axis
  ##
  ## Input:
  ##   - A tensor
  ##   - An axis (int)
  ##
  ## Returns:
  ##   - A tuple of tensors (indices, min) along this axis
  ##
  ## Example:
  ##   .. code:: nim
  ##     let a = [[0, 4, 7],
  ##              [1, 9, 5],
  ##              [3, 4, 1]].toTensor
  ##     assert argmin(a, 0).indices == [[0, 0, 2]].toTensor
  ##     assert argmin(a, 1).indices == [[0],
  ##                                     [0],
  ##                                     [2]].toTensor

  if arg.size == 0:
    result.indices.reset()
    result.mins.reset()

  result.mins = arg.atAxisIndex(axis, 0).clone()
  result.indices = zeros[int](result.mins.shape)

  let dmin = result.mins.unsafe_raw_buf()
  let dind = result.indices.unsafe_raw_buf()

  for i, subtensor in arg.enumerateAxis(axis, 1, arg.shape[axis] - 1):
    for j, val in enumerate(subtensor):
      if val < dmin[j]:
        dind[j] = i
        dmin[j] = val

proc argmin*[T](arg: Tensor[T], axis: int): Tensor[int] {.inline.}=
  ## Returns the index of the minimum along an axis
  ##
  ## Input:
  ##   - A tensor
  ##   - An axis (int)
  ##
  ## Returns:
  ##   - A tensor of index of the minimums along this axis
  ##
  ## Example:
  ##   .. code:: nim
  ##     let a = [[0, 4, 7],
  ##              [1, 9, 5],
  ##              [3, 4, 1]].toTensor
  ##     assert argmin(a, 0) == [[2, 1, 0]].toTensor
  ##     assert argmin(a, 1) == [[2],
  ##                             [1],
  ##                             [1]].toTensor
  argmin_min(arg, axis).indices

proc percentile*[T](arg: Tensor[T], p: int, isSorted = false): float =
  ## statistical percentile value of ``t``, where ``p`` percentile value
  ## is between ``0`` and ``100`` inclusively,
  ## and ``p=0`` gives the min value, ``p=100`` gives the max value
  ## and ``p=50`` gives the median value.
  ##
  ## If the input percentile does not match an element of `t` exactly
  ## the result is the linear interpolation between the neighbors.
  ##
  ## ``t`` does not need to be sorted, because ``percentile`` sorts
  ## a copy of the data itself. If ``isSorted`` is ``true`` however,
  ## no sorting is done.
  # TODO: we could in principle also return `T`, but then we cannot do
  # interpolation between values. Hm.
  if arg.size == 0: result = 0.0
  elif p <= 0: result = min(arg).float
  elif p >= 100: result = max(arg).float
  else:
    let a = if not isSorted: sorted(arg.reshape([1, arg.size]).squeeze) else: arg
    let f = (arg.size - 1) * p / 100
    let i = floor(f).int
    if f == i.float: result = a[i].float
    else:
      # interpolate linearly
      let frac = f - i.float
      result = (a[i].float + (a[i+1] - a[i]).float * frac)

proc median*[T](arg: Tensor[T], isSorted = false): float {.inline.} =
  ## Compute the median of all elements (same as `arg.percentile(50)`)
  percentile(arg, 50, isSorted)

proc iqr*[T](arg: Tensor[T]): float =
  ## Returns the interquartile range of the 1D tensor `t`.
  ##
  ## The interquartile range (IQR) is the distance between the
  ## 25th and 75th percentile
  let tS = arg.sorted
  result = percentile(tS, 75, isSorted = true) -
           percentile(tS, 25, isSorted = true)

proc cumsum*[T](arg: Tensor[T], axis: int = 0): Tensor[T] = # from hugogranstrom
  ## Calculates the cumulative sum of a rank-n Tensor.
  ## Inputs:
  ##  - t: a rank-n tensor to cumulatively sum
  ##  - axis: int
  ## Returns:
  ##  - A tensor cumulatively summed at axis, that is, add each value to
  mixin `_`
  result = zeros_like(arg)
  for i, tAxis in enumerateAxis(arg, axis):
    var temp = result.atAxisIndex(axis, i)
    if i == 0:
      temp[_] = tAxis
    else:
      temp[_] = result.atAxisIndex(axis, i-1) + tAxis

proc cumprod*[T](arg: Tensor[T], axis: int = 0): Tensor[T] = # from hugogranstrom
  ## Calculates the cumulative sum of a rank-n Tensor.
  ## Inputs:
  ##  - t: a rank-n tensor to cumulatively sum
  ##  - axis: int
  ## Returns:
  ##  - A tensor cumulatively summed at axis, that is, add each value to
  mixin `_`
  result = zeros_like(arg)
  for i, tAxis in enumerateAxis(arg, axis):
    var temp = result.atAxisIndex(axis, i)
    if i == 0:
      temp[_] = tAxis
    else:
      temp[_] = result.atAxisIndex(axis, i-1) *. tAxis

proc diff*[T](arg: Tensor[T], n=1, axis: int = -1): Tensor[T] =
  ## Calculate the n-th discrete difference along the given axis.
  ##
  ## The first difference is given by out[i] = a[i+1] - a[i] along the given axis.
  ## Higher differences are calculated by using diff recursively.
  ##
  ## Input:
  ##   - A tensor
  ##   - n: The number of times values are differenced.
  ##        If zero, the input is returned as-is.
  ##   - axis: The axis along which the difference is taken,
  ##           default is the last axis.
  mixin `_`
  assert n >= 0, "diff order (" & $n & ") cannot be negative"
  if n == 0 or arg.size == 0:
    return arg
  let axis = if axis == -1:
    arg.shape.len + axis
  else:
    axis
  assert axis < arg.shape.len,
    "diff axis (" & $axis & ") cannot be greater than input shape length (" & $arg.shape.len & ")"
  var result_shape = arg.shape
  result_shape[axis] -= 1
  result = zeros[T](result_shape)
  for i, tAxis in enumerateAxis(arg, axis):
    if unlikely(i == 0):
      continue
    var temp = result.atAxisIndex(axis, i-1)
    when T is bool:
      temp[_] = tAxis != arg.atAxisIndex(axis, i-1)
    else:
      temp[_] = tAxis -. arg.atAxisIndex(axis, i-1)
  if n > 1:
    result = diff(result, n=n-1, axis=axis)

proc unwrap_period*[T: SomeNumber](t: Tensor[T], discont: T = -1, axis = -1, period: T = default(T)): Tensor[T] {.noinit.} =
    # Unwrap a tensor by taking the complement of large deltas with respect to a period.
    #
    # This unwraps a tensor `t` by changing elements which have an absolute
    # difference from their predecessor of more than ``max(discont, period/2)``
    # to their `period`-complementary values.
    #
    # For the default case where `period` is `2*PI` and `discont` is
    # `PI`, this unwraps a radian phase `t` such that adjacent differences
    # are never greater than `PI` by adding `2*k*PIi` for some integer `k`.
    #
    # Inputs:
    #   - t: Input Tensor.
    #   - discont: Maximum discontinuity between values. Default is `period/2`.
    #       Values below `period/2` are treated as if they were `period/2`.
    #       To have an effect different than the default, `discont` must be
    #       larger than `period/2`.
    #   - axis: Axis along which the unwrap will be done. Default is the last axis.
    #   - period: Size of the range over which the input wraps.
    #             By default, it is ``2*PI``.
    # Return:
    #   - Output Tensor.
    #
    # Notes:
    #   - If the discontinuity in `t` is smaller than ``period/2``,
    #   but larger than `discont`, no unwrapping is done because taking
    #   the complement would only make the discontinuity larger.
    #   - The code in this function is heavily based upon and equivalent
    #   to numpy's `unwrap()` function.
  mixin `_`
  let axis = if axis == -1:
    t.shape.len + axis
  else:
    axis
  let td = t.diff(axis=axis)
  let period: T = if period == default(T):
    when T is int:
      raise newException(ValueError, "unwrap period must be specified for integer types")
    else:
      2.0 * PI
  else:
    period
  let discont = if discont == -1:
    T(period/2)
  else:
    discont
  when T is int:
    when (NimMajor, NimMinor, NimPatch) >= (2, 0, 0):
      let (interval_high, rem) = divmod(period, 2)
    else:
      let interval_high = period div 2
      let rem = period mod 2
    let boundary_ambiguous = rem == 0
  else:
    let interval_high = period / 2
    let boundary_ambiguous = true
  let interval_low = -interval_high
  var tdmod = (td -. interval_low).floorMod(period) +. interval_low
  if boundary_ambiguous:
    const zero: T = T(0)
    tdmod[(tdmod ==. interval_low) and (td >. zero)] = interval_high
  var ph_correct = tdmod - td
  ph_correct[abs(td) <. discont] = 0
  result = t.clone()
  let ph_correct_cumsum = ph_correct.cumsum(axis)
  if t.rank == 1:
    result[1.._] = t[1.._] +. ph_correct_cumsum
  else:
    for i, tAxis in enumerateAxis(t, axis):
      if unlikely(i < 1):
        continue
      let pAxis = ph_correct_cumsum.atAxisIndex(axis, i-1)
      var temp = result.atAxisIndex(axis, i)
      temp[_] = tAxis +. pAxis

when (NimMajor, NimMinor, NimPatch) > (1, 6, 0):
  import std/atomics
proc nonzero*[T](arg: Tensor[T]): Tensor[int] =
  ## Returns the indices, which are non zero as a `Tensor[int]`.
  ##
  ## The resulting tensor is 2 dimensional and has one element for each
  ## dimension in ``t``. Each of those elements contains the indicies along
  ## the corresponding axis (element 0 == axis 0), which are non zero.
  ##
  ## Input:
  ##   - A tensor
  ##
  ## Returns:
  ##   - A 2D tensor with N elements, where N is the rank of ``t``
  ##
  ## Example:
  ##   .. code:: nim
  ##      let a = [[3, 0, 0],
  ##               [0, 4, 0],
  ##               [5, 6, 0]].toTensor()
  ##      assert a.nonzero == [[0, 1, 2, 2], [0, 1, 0, 1]].toTensor
  ##      #                    ^-- indices.. ^ ..for  axis 0
  ##      #                                  ∟-- indices for axis 1
  ##      # axis 0: [0, 1, 2, 2] refers to:
  ##      # - 0 -> 3 in row 0
  ##      # - 1 -> 4 in row 1
  ##      # - 2 -> 5 in row 2
  ##      # - 2 -> 6 in row 2
  ##      # axis 1: [0, 1, 0, 1] refers to:
  ##      # - 0 -> 3 in col 0
  ##      # - 1 -> 4 in col 1
  ##      # - 0 -> 5 in col 0
  ##      # - 1 -> 6 in col 1
  when (NimMajor, NimMinor, NimPatch) > (1, 6, 0):
    ## Use `Atomic` counter. If compiled with `-d:openmp` otherwise the code breaks!
    var count: Atomic[int]
    count.store(0)
    let mask = map_inline(arg):
      block:
        let cond = x != 0.T
        if cond:
          atomicInc count
        cond

    result = newTensor[int]([arg.shape.len, count.load])
  else:
    let mask = map_inline(arg): # generate the mask
      x != 0.T
    var count = 0 # and count non zero elements (avoid openmp issues)
    for x in mask:
      if x:
        inc count
    result = newTensor[int]([arg.shape.len, count])

  var ax = 0 # current axis
  var k = 0 # counter for indices in one axis
  for idx, x in mask:
    if x:
      ax = 0
      for j in idx:
        result[ax, k] = j
        inc ax
      inc k
