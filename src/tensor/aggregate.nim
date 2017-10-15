# Copyright 2017 Mamy André-Ratsimbazafy
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
        ./higher_order,
        ./operators_blas_l1,
        ./math_functions,
        math

# ### Standard aggregate functions
# TODO consider using stats from Nim standard lib: https://nim-lang.org/docs/stats.html#standardDeviation,RunningStat

proc sum*[T](t: Tensor[T]): T =
  ## Compute the sum of all elements
  t.reduce_inline():
    x+=y

proc sum*[T](t: Tensor[T], axis: int): Tensor[T] {.noInit,noInit.} =
  ## Compute the sum of all elements along an axis
  t.reduce_axis_inline(axis):
    x+=y

proc product*[T](t: Tensor[T]): T {.inline.}=
  ## Compute the product of all elements
  t.reduce_inline():
    x*=y

proc product*[T](t: Tensor[T], axis: int): Tensor[T] {.noInit,inline.}=
  ## Compute the product along an axis
  t.reduce_axis_inline(axis):
    x.melwise_mul(y)

proc mean*[T: SomeInteger](t: Tensor[T]): T {.inline.}=
  ## Compute the mean of all elements
  t.sum div t.size.T

proc mean*[T: SomeInteger](t: Tensor[T], axis: int): Tensor[T] {.noInit,inline.}=
  ## Compute the mean along an axis
  t.sum(axis) div t.shape[axis].T

proc mean*[T: SomeReal](t: Tensor[T]): T {.inline.}=
  ## Compute the mean of all elements
  t.sum / t.size.T

proc mean*[T: SomeReal](t: Tensor[T], axis: int): Tensor[T] {.noInit,inline.}=
  ## Compute the mean along an axis
  t.sum(axis) / t.shape[axis].T

proc min*[T](t: Tensor[T]): T =
  ## Compute the min of all elements
  t.reduce_inline():
    x = min(x,y)

proc min*[T](t: Tensor[T], axis: int): Tensor[T] {.noInit.} =
  ## Compute the min along an axis
  t.reduce_axis_inline(axis):
    for ex, ey in mzip(x,y):
      ex = min(ex,ey)

proc max*[T](t: Tensor[T]): T =
  ## Compute the max of all elements
  t.reduce_inline():
    x = max(x,y)

proc max*[T](t: Tensor[T], axis: int): Tensor[T] {.noInit.} =
  ## Compute the max along an axis
  t.reduce_axis_inline(axis):
    for ex, ey in mzip(x,y):
      ex = max(ex,ey)

proc variance*[T: SomeReal](t: Tensor[T]): T =
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

proc variance*[T: SomeReal](t: Tensor[T], axis: int): Tensor[T] {.noInit.} =
  ## Compute the variance of all elements
  ## The normalization is by the (n-1), like in the formal definition
  let mean = t.mean(axis)
  result = t.fold_axis_inline(axis) do:
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

proc std*[T: SomeReal](t: Tensor[T]): T {.inline.} =
  ## Compute the standard deviation of all elements
  ## The normalization is by the (n-1), like in the formal definition
  sqrt(t.variance())

proc std*[T: SomeReal](t: Tensor[T], axis: int): Tensor[T] {.noInit,inline.} =
  ## Compute the standard deviation of all elements
  ## The normalization is by the (n-1), like in the formal definition
  sqrt(t.variance(axis))
