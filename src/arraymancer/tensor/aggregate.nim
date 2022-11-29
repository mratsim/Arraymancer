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
        ./higher_order_applymap,
        ./math_functions,
        ./accessors,
        ./algorithms,
        ./private/p_empty_tensors,
        math

import complex except Complex64, Complex32

# ### Standard aggregate functions
# TODO consider using stats from Nim standard lib: https://nim-lang.org/docs/stats.html#standardDeviation,RunningStat

# Note: for aggregate that returns scalar, if the tensor is empty,
#       Numpy seems to return the neutral element, do we want that?


proc sum*[T](t: Tensor[T]): T =
  ## Compute the sum of all elements
  t.reduce_inline():
    x+=y

proc sum*[T](t: Tensor[T], axis: int): Tensor[T] {.noinit.} =
  ## Compute the sum of all elements along an axis
  returnEmptyIfEmpty(t)
  t.reduce_axis_inline(axis):
    x+=y

proc product*[T](t: Tensor[T]): T =
  ## Compute the product of all elements
  t.reduce_inline():
    x*=y

proc product*[T](t: Tensor[T], axis: int): Tensor[T] {.noinit.}=
  ## Compute the product along an axis
  returnEmptyIfEmpty(t)
  t.reduce_axis_inline(axis):
    x.melwise_mul(y)

proc mean*[T: SomeInteger](t: Tensor[T]): T {.inline.}=
  ## Compute the mean of all elements
  ##
  ## Warning ⚠: Since input is integer, output will also be integer (using integer division)
  t.sum div t.size.T

proc mean*[T: SomeInteger](t: Tensor[T], axis: int): Tensor[T] {.noinit,inline.}=
  ## Compute the mean along an axis
  ##
  ## Warning ⚠: Since input is integer, output will also be integer (using integer division)
  returnEmptyIfEmpty(t)
  t.sum(axis) div t.shape[axis].T

proc mean*[T: SomeFloat](t: Tensor[T]): T {.inline.}=
  ## Compute the mean of all elements
  t.sum / t.size.T

proc mean*[T: Complex[float32] or Complex[float64]](t: Tensor[T]): T {.inline.}=
  ## Compute the mean of all elements
  type F = T.T # Get float subtype of Complex[T]
  t.sum / complex(t.size.F, 0.F)

proc mean*[T: SomeFloat](t: Tensor[T], axis: int): Tensor[T] {.noinit,inline.}=
  ## Compute the mean along an axis
  returnEmptyIfEmpty(t)
  t.sum(axis) / t.shape[axis].T

proc mean*[T: Complex[float32] or Complex[float64]](t: Tensor[T], axis: int): Tensor[T] {.noinit,inline.}=
  ## Compute the mean along an axis
  returnEmptyIfEmpty(t)
  type F = T.T # Get float subtype of Complex[T]
  t.sum(axis) / complex(t.shape[axis].F, 0.F)

proc min*[T](t: Tensor[T]): T =
  ## Compute the min of all elements
  reduce_inline(t):
    x = min(x,y)

proc min*[T](t: Tensor[T], axis: int): Tensor[T] {.noinit.} =
  ## Compute the min along an axis
  returnEmptyIfEmpty(t)
  t.reduce_axis_inline(axis):
    for ex, ey in mzip(x,y):
      ex = min(ex,ey)

proc max*[T](t: Tensor[T]): T =
  ## Compute the max of all elements
  t.reduce_inline():
    x = max(x,y)

proc max*[T](t: Tensor[T], axis: int): Tensor[T] {.noinit.} =
  ## Compute the max along an axis
  returnEmptyIfEmpty(t)
  t.reduce_axis_inline(axis):
    for ex, ey in mzip(x,y):
      ex = max(ex,ey)

proc variance*[T: SomeFloat](t: Tensor[T]): T =
  ## Compute the sample variance of all elements
  ## The normalization is by (n-1), also known as Bessel's correction,
  ## which partially correct the bias of estimating a population variance from a sample of this population.
  let mean = t.mean()
  result = t.fold_inline() do:
    # Initialize to the first element
    x = square(y - mean)
  do:
    # Fold in parallel by summing remaning elements
    x += square(y - mean)
  do:
    # Merge parallel folds
    x += y
  result /= (t.size-1).T

proc variance*[T: SomeFloat](t: Tensor[T], axis: int): Tensor[T] {.noinit.} =
  ## Compute the variance of all elements
  ## The normalization is by the (n-1), like in the formal definition
  returnEmptyIfEmpty(t)
  let mean = t.mean(axis)
  result = t.fold_axis_inline(Tensor[T], axis) do:
    # Initialize to the first element
    x = square(y - mean)
  do:
    # Fold in parallel by summing remaning elements
    for ex, ey, em in mzip(x,y,mean):
      ex += square(ey - em)
  do:
    # Merge parallel folds
    x += y
  result /= (t.shape[axis]-1).T

proc std*[T: SomeFloat](t: Tensor[T]): T {.inline.} =
  ## Compute the standard deviation of all elements
  ## The normalization is by the (n-1), like in the formal definition
  sqrt(t.variance())

proc std*[T: SomeFloat](t: Tensor[T], axis: int): Tensor[T] {.noinit,inline.} =
  ## Compute the standard deviation of all elements
  ## The normalization is by the (n-1), like in the formal definition
  returnEmptyIfEmpty(t)
  sqrt(t.variance(axis))

proc argmax_max*[T: SomeNumber](t: Tensor[T], axis: int): tuple[indices: Tensor[int], maxes: Tensor[T]] {.noinit.} =
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

  if t.size == 0:
    result.indices.reset()
    result.maxes.reset()

  result.maxes = t.atAxisIndex(axis, 0).clone()
  result.indices = zeros[int](result.maxes.shape)

  let dmax = result.maxes.unsafe_raw_buf()
  let dind = result.indices.unsafe_raw_buf()

  for i, subtensor in t.enumerateAxis(axis, 1, t.shape[axis] - 1):
    for j, val in enumerate(subtensor):
      if val > dmax[j]:
        dind[j] = i
        dmax[j] = val

proc argmax*[T](t: Tensor[T], axis: int): Tensor[int] {.inline.}=
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
  argmax_max(t, axis).indices

proc argmin_min*[T: SomeNumber](t: Tensor[T], axis: int): tuple[indices: Tensor[int], mins: Tensor[T]] {.noInit.} =
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

  if t.size == 0:
    result.indices.reset()
    result.mins.reset()

  result.mins = t.atAxisIndex(axis, 0).clone()
  result.indices = zeros[int](result.mins.shape)

  let dmin = result.mins.unsafe_raw_buf()
  let dind = result.indices.unsafe_raw_buf()

  for i, subtensor in t.enumerateAxis(axis, 1, t.shape[axis] - 1):
    for j, val in enumerate(subtensor):
      if val < dmin[j]:
        dind[j] = i
        dmin[j] = val

proc argmin*[T](t: Tensor[T], axis: int): Tensor[int] {.inline.}=
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
  argmin_min(t, axis).indices

proc percentile*[T](t: Tensor[T], p: int, isSorted = false): float =
  ## statistical percentile value of ``t``, where ``p`` percentile value
  ## is between ``0`` and ``100`` inclusively,
  ## and ``p=0`` gives the min value, ``p=100`` gives the max value
  ## and ``p=50`` gives the median value.
  ##
  ## If the input percentile does not match an element of `t` exactly
  ## the result is the linear interpolation between the neighbors.
  ##
  ## ``t`` does not need to be sorted, because ``percentile`` sorts
  ## a copy of the data itself. If ``isSorted``` is ``true`` however,
  ## no sorting is done.
  # TODO: we could in principle also return `T`, but then we cannot do
  # interpolation between values. Hm.
  if t.size == 0: result = 0.0
  elif p <= 0: result = min(t).float
  elif p >= 100: result = max(t).float
  else:
    let a = if not isSorted: sorted(t) else: t
    let f = (t.size - 1) * p / 100
    let i = floor(f).int
    if f == i.float: result = a[i].float
    else:
      # interpolate linearly
      let frac = f - i.float
      result = (a[i].float + (a[i+1] - a[i]).float * frac)

proc iqr*[T](t: Tensor[T]): float =
  ## Returns the interquartile range of the 1D tensor `t`.
  ##
  ## The interquartile range (IQR) is the distance between the
  ## 25th and 75th percentile
  let tS = t.sorted
  result = percentile(tS, 75, isSorted = true) -
           percentile(tS, 25, isSorted = true)

proc cumsum*[T](t: Tensor[T], axis: int = 0): Tensor[T] = # from hugogranstrom
  ## Calculates the cumulative sum of a rank-n Tensor.
  ## Inputs:
  ##  - t: a rank-n tensor to cumulatively sum
  ##  - axis: int
  ## Returns:
  ##  - A tensor cumulatively summed at axis, that is, add each value to
  mixin `_`
  result = zeros_like(t)
  for i, tAxis in enumerateAxis(t, axis):
    var temp = result.atAxisIndex(axis, i)
    if i == 0:
      temp[_] = tAxis
    else:
      temp[_] = result.atAxisIndex(axis, i-1) + tAxis

proc cumprod*[T](t: Tensor[T], axis: int = 0): Tensor[T] = # from hugogranstrom
  ## Calculates the cumulative sum of a rank-n Tensor.
  ## Inputs:
  ##  - t: a rank-n tensor to cumulatively sum
  ##  - axis: int
  ## Returns:
  ##  - A tensor cumulatively summed at axis, that is, add each value to
  mixin `_`
  result = zeros_like(t)
  for i, tAxis in enumerateAxis(t, axis):
    var temp = result.atAxisIndex(axis, i)
    if i == 0:
      temp[_] = tAxis
    else:
      temp[_] = result.atAxisIndex(axis, i-1) *. tAxis

proc nonzero*[T](t: Tensor[T]): Tensor[int] =
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
  ##      #                                  |-- indices for axis 1
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
  var count = 0 # number of non zero elements
  let mask = map_inline(t):
    block:
      let cond = x != 0.T
      if cond:
        inc count
      cond

  result = newTensor[int]([t.shape.len, count])
  var ax = 0 # current axis
  var k = 0 # counter for indices in one axis
  for idx, x in mask:
    if x:
      ax = 0
      for j in idx:
        result[ax, k] = j
        inc ax
      inc k
