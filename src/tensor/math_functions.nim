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
        ./init_cpu,
        ./higher_order,
        ./ufunc

# Non-operator math functions

proc elwise_mul*[T](a, b: Tensor[T]): Tensor[T] {.noInit.} =
  ## Element-wise multiply
  map2_inline(a, b, x * y)

proc melwise_mul*[T](a: var Tensor[T], b: Tensor[T]) =
  ## Element-wise multiply
  a.apply2_inline(b, x * y)

proc elwise_div*[T: Someinteger](a, b: Tensor[T]): Tensor[T] {.noInit.} =
  ## Element-wise division
  map2_inline(a, b, x div y)

proc elwise_div*[T: SomeReal](a, b: Tensor[T]): Tensor[T] {.noInit.} =
  ## Element-wise division
  map2_inline(a, b, x / y)

proc melwise_div*[T: Someinteger](a: var Tensor[T], b: Tensor[T]) =
  ## Element-wise division (in-place)
  a.apply2_inline(b, x div y)

proc melwise_div*[T: SomeReal](a: var Tensor[T], b: Tensor[T]) =
  ## Element-wise division (in-place)
  a.apply2_inline(b, x / y)

proc reciprocal*[T: SomeReal](t: Tensor[T]): Tensor[T] {.noInit.} =
  ## Return a tensor with the reciprocal 1/x of all elements
  t.map_inline(1.T/x)

proc mreciprocal*[T: SomeReal](t: var Tensor[T]) =
  ## Apply the reciprocal 1/x in-place to all elements of the Tensor
  t.apply_inline(1.T/x)

proc negate*[T: SomeSignedInt|SomeReal](t: Tensor[T]): Tensor[T] {.noInit.} =
  ## Return a tensor with all elements negated (10 -> -10)
  t.map_inline(-x)

proc mnegate*[T: SomeSignedInt|SomeReal](t: var Tensor[T]) =
  ## Negate in-place all elements of the tensor (10 -> -10)
  t.apply_inline(-x)

proc `-`*[T: SomeNumber](t: Tensor[T]): Tensor[T] {.noInit.} =
  ## Negate all values of a Tensor
  t.map_inline(-x)

# Built-in nim function that doesn't work with makeUniversal
proc abs*[T](t: Tensor[T]): Tensor[T] {.noInit.} =
  ## Return a Tensor with absolute values of all elements
  t.map_inline(abs(x))

proc mabs*[T](t: Tensor[T]): Tensor[T] {.noInit.} =
  ## Return a Tensor with absolute values of all elements
  t.apply_inline(abs(x))

proc clamp*[T](t: Tensor[T], min, max: T): Tensor[T] {.noInit.} =
  t.map_inline(clamp(x, min, max))

proc mclamp*[T](t: var Tensor[T], min, max: T) =
  t.apply_inline(clamp(x, min, max))

proc square*[T](x: T): T {.inline.} =
  ## Return x*x
  x*x

makeUniversal(square)

proc absolute_error*[T](y_true, y: T): T {.inline.} =
  ## Absolute error for a single value, |y_true - y|
  result = abs(y_true - y)

proc absolute_error*[T](y_true, y: Tensor[T]): Tensor[T] {.noInit.} =
  ## Element-wise absolute error for a tensor
  result = map2_inline(y_true, y, absolute_error(x,y))

proc mean_absolute_error*[T](y_true, y: Tensor[T]): T =
  ## Also known as L1 loss, absolute error between elements:
  ## sum(|y_true - y|)/m
  ## where m is the number of elements
  result = map2_inline(y_true, y, absolute_error(x,y)).mean()

proc squared_error*[T](y_true, y: T): T {.inline.} =
  ## Squared error for a single value, |y_true - y|^2
  result = square(y_true - y)

proc squared_error*[T](y_true, y: Tensor[T]): Tensor[T] {.noInit.} =
  ## Element-wise squared error for a tensor, |y_true - y|^2
  result = map2_inline(y_true, y, squared_error(x,y))

proc mean_squared_error*[T](y_true, y: Tensor[T]): T =
  ## Also known as MSE or L2 loss, mean squared error between elements:
  ## sum(|y_true - y|^2)/m
  ## where m is the number of elements
  result = squared_error(y_true, y).mean()

proc relative_error*[T](y_true, y: T): T {.inline.} =
  ## Relative error, |y_true - y|/max(|y_true|, |y|)
  ## Normally the relative error is defined as |y_true - y| / |y_true|,
  ## but here max is used to make it symmetric and to prevent dividing by zero,
  ## guaranteed to return zero in the case when both values are zero.
  let denom = max(abs(y_true), abs(y))
  if denom == 0.T:
    return 0.T
  result = abs(y_true - y) / denom

proc relative_error*[T](y_true, y: Tensor[T]): Tensor[T] {.noInit.} =
  ## Relative error for Tensor, element-wise |y_true - x|/max(|y_true|, |x|)
  ## Normally the relative error is defined as |y_true - x| / |y_true|,
  ## but here max is used to make it symmetric and to prevent dividing by zero,
  ## guaranteed to return zero in the case when both values are zero.
  result = map2_inline(y_true, y, relative_error(x,y))

proc mean_relative_error*[T](y_true, y: Tensor[T]): T =
  ## Mean relative error for Tensor, mean of the element-wise
  ## |y_true - y|/max(|y_true|, |y|)
  ## Normally the relative error is defined as |y_true - y| / |y_true|,
  ## but here max is used to make it symmetric and to prevent dividing by zero,
  ## guaranteed to return zero in the case when both values are zero.
  result = relative_error(y_true, y).mean()

proc numerical_gradient*[T](input: T, f: (proc(x: T): T), h: T = 1e-5.T): T {.inline.} =
  ## Compute numerical gradient for any function w.r.t. to an input value,
  ## useful for gradient checking, recommend using float64 types to assure
  ## numerical precision. The gradient is calculated as:
  ## (f(x + h) - f(x - h)) / (2*h)
  ## where h is a small number, typically 1e-5.
  result = (f(input + h) - f(input - h)) / (2.0.T * h)

proc numerical_gradient*[T](input: Tensor[T], f: (proc(x: Tensor[T]): T), h: T = 1e-5.T): Tensor[T] {.noInit.} =
  ## Compute numerical gradient for any function w.r.t. to an input Tensor,
  ## useful for gradient checking, recommend using float64 types to assure
  ## numerical precision. The gradient is calculated as:
  ## (f(x + h) - f(x - h)) / (2*h)
  ## where h is a small number, typically 1e-5
  ## f(x) will be called for each input elements with +h and -h pertubation.
  # Iterate over all elements calculating each partial derivative

  # Note: If T: float32, Nim may compile but produce incompatible type in C code
  result = newTensorUninit[T](input.shape)
  var x = input
  for i, val in x.menumerate:
    let orig_val = val
    val = orig_val + h
    let fa = f(x)
    val = orig_val - h
    let fb = f(x)
    val = orig_val
    result.data[i] = (fa - fb) / (2.0.T * h)
