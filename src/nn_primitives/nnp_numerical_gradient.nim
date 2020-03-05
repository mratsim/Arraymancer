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

import ../tensor/tensor

proc numerical_gradient*[T](input: T, f: (proc(x: T): T), h = T(1e-5)): T {.inline.} =
  ## Compute numerical gradient for any function w.r.t. to an input value,
  ## useful for gradient checking, recommend using float64 types to assure
  ## numerical precision. The gradient is calculated as:
  ## (f(x + h) - f(x - h)) / (2*h)
  ## where h is a small number, typically 1e-5.
  result = (f(input + h) - f(input - h)) / (2.0.T * h)

proc numerical_gradient*[T](input: Tensor[T], f: (proc(x: Tensor[T]): T), h = T(1e-5)): Tensor[T] {.noInit.} =
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
    result.unsafe_raw_buf[i] = (fa - fb) / (2.0.T * h)
