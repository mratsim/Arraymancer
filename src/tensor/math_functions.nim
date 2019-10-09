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
        ./higher_order_applymap,
        ./ufunc
import complex except Complex64, Complex32

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

proc elwise_div*[T: SomeFloat](a, b: Tensor[T]): Tensor[T] {.noInit.} =
  ## Element-wise division
  map2_inline(a, b, x / y)

proc melwise_div*[T: Someinteger](a: var Tensor[T], b: Tensor[T]) =
  ## Element-wise division (in-place)
  a.apply2_inline(b, x div y)

proc melwise_div*[T: SomeFloat](a: var Tensor[T], b: Tensor[T]) =
  ## Element-wise division (in-place)
  a.apply2_inline(b, x / y)

proc reciprocal*[T: SomeFloat|Complex[float32]|Complex[float64]](t: Tensor[T]): Tensor[T] {.noInit.} =
  ## Return a tensor with the reciprocal 1/x of all elements
  t.map_inline(1.T/x)

proc mreciprocal*[T: SomeFloat|Complex[float32]|Complex[float64]](t: var Tensor[T]) =
  ## Apply the reciprocal 1/x in-place to all elements of the Tensor
  t.apply_inline(1.T/x)

proc negate*[T: SomeSignedInt|SomeFloat](t: Tensor[T]): Tensor[T] {.noInit.} =
  ## Return a tensor with all elements negated (10 -> -10)
  t.map_inline(-x)

proc mnegate*[T: SomeSignedInt|SomeFloat](t: var Tensor[T]) =
  ## Negate in-place all elements of the tensor (10 -> -10)
  t.apply_inline(-x)

proc `-`*[T: SomeNumber](t: Tensor[T]): Tensor[T] {.noInit.} =
  ## Negate all values of a Tensor
  t.map_inline(-x)

# Built-in nim function that doesn't work with makeUniversal
proc abs*[T:SomeNumber](t: Tensor[T]): Tensor[T] {.noInit.} =
  ## Return a Tensor with absolute values of all elements
  t.map_inline(abs(x))

# complex abs -> float
proc abs*(t: Tensor[Complex[float64]]): Tensor[float64] {.noInit.} =
  ## Return a Tensor with absolute values of all elements
  t.map_inline(abs(x))

proc abs*(t: Tensor[Complex[float32]]): Tensor[float32] {.noInit.} =
  ## Return a Tensor with absolute values of all elements
  t.map_inline(abs(x))

proc mabs*[T](t: var Tensor[T]) =
  ## Return a Tensor with absolute values of all elements
  # FIXME: how to inplace convert Tensor[Complex] to Tensor[float]
  t.apply_inline(abs(x))

proc clamp*[T](t: Tensor[T], min, max: T): Tensor[T] {.noInit.} =
  t.map_inline(clamp(x, min, max))

proc mclamp*[T](t: var Tensor[T], min, max: T) =
  t.apply_inline(clamp(x, min, max))

proc square*[T](x: T): T {.inline.} =
  ## Return x*x
  x*x

makeUniversal(square)
