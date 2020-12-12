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

import ../../tensor

# ####################################################

proc squared_error*[T](y, y_true: T): T {.inline.} =
  ## Squared error for a single value, |y_true - y| ^2
  result = square(y_true - y)

proc squared_error*[T](y, y_true: Tensor[T]): Tensor[T] {.noInit.} =
  ## Element-wise squared error for a tensor, |y_true - y| ^2
  result = newTensorUninit[T](y.shape)
  forEach r in result,
          testval in y,
          truthval in y_true:
    r = squared_error(testval,truthval)

proc mean_squared_error*[T](y, y_true: Tensor[T]): T =
  ## Also known as MSE or L2 loss, mean squared error between elements:
  ## sum(|y_true - y| ^2)/m
  ## where m is the number of elements
  result = y.squared_error(y_true).mean()

# ####################################################

proc relative_error*[T: SomeFloat](y, y_true: T): T {.inline.} =
  ## Relative error, |y_true - y|/max(|y_true|, |y|)
  ## Normally the relative error is defined as |y_true - y| / |y_true|,
  ## but here max is used to make it symmetric and to prevent dividing by zero,
  ## guaranteed to return zero in the case when both values are zero.
  # We require float (and not complex as complex.abs will cause issue)
  let denom = max(abs(y_true), abs(y))
  if denom == 0.T:
    return 0.T
  result = abs(y_true - y) / denom

proc relative_error*[T](y, y_true: Tensor[T]): Tensor[T] {.noInit.} =
  ## Relative error for Tensor, element-wise |y_true - x|/max(|y_true|, |x|)
  ## Normally the relative error is defined as |y_true - x| / |y_true|,
  ## but here max is used to make it symmetric and to prevent dividing by zero,
  ## guaranteed to return zero in the case when both values are zero.
  result = newTensorUninit[T](y.shape)
  forEach r in result,
          testval in y,
          truthval in y_true:
    r = relative_error(testval,truthval)

proc mean_relative_error*[T](y, y_true: Tensor[T]): T =
  ## Mean relative error for Tensor, mean of the element-wise
  ## |y_true - y|/max(|y_true|, |y|)
  ## Normally the relative error is defined as |y_true - y| / |y_true|,
  ## but here max is used to make it symmetric and to prevent dividing by zero,
  ## guaranteed to return zero in the case when both values are zero.
  result = y.relative_error(y_true).mean()

# ####################################################

proc absolute_error*[T: SomeFloat](y, y_true: T): T {.inline.} =
  ## Absolute error for a single value, |y_true - y|
  # We require float (and not complex as complex.abs will cause issue)
  result = abs(y_true - y)

proc absolute_error*[T](y, y_true: Tensor[T]): Tensor[T] {.noInit.} =
  ## Element-wise absolute error for a tensor
  result = newTensorUninit[T](y.shape)
  forEach r in result,
          testval in y,
          truthval in y_true:
    r = absolute_error(testval,truthval)

proc mean_absolute_error*[T](y, y_true: Tensor[T]): T =
  ## Also known as L1 loss, absolute error between elements:
  ## sum(|y_true - y|)/m
  ## where m is the number of elements
  result = y.absolute_error(y_true).mean()
