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

proc `.+`*[T: SomeNumber](a, b: Tensor[T]): Tensor[T] {.noSideEffect, inline.} =
  ## Broadcasted addition for tensors of incompatible but broadcastable shape
  let (tmp_a, tmp_b) = unsafeBroadcast2(a, b)
  return tmp_a + tmp_b

proc `.+`*[T: SomeNumber](val: T, t: Tensor[T]): Tensor[T] {.noSideEffect, inline.} =
  ## Broadcasted addition for tensor + scalar
  proc f(x: T): T = x + val
  return t.map(f)

proc `.+`*[T: SomeNumber](t: Tensor[T], val: T): Tensor[T] {.noSideEffect, inline.} =
  ## Broadcasted addition for scalar + tensor
  proc f(x: T): T = x + val
  return t.map(f)

proc `.-`*[T: SomeNumber](a, b: Tensor[T]): Tensor[T] {.noSideEffect, inline.} =
  ## Broadcasted addition for tensors of incompatible but broadcastable shape
  let (tmp_a, tmp_b) = unsafeBroadcast2(a, b)
  return tmp_a - tmp_b

proc `.-`*[T: SomeNumber](val: T, t: Tensor[T]): Tensor[T] {.noSideEffect, inline.} =
  ## Broadcasted substraction for tensor - scalar
  proc f(x: T): T = val - x
  return t.map(f)

proc `.-`*[T: SomeNumber](t: Tensor[T], val: T): Tensor[T] {.noSideEffect, inline.} =
  ## Broadcasted substraction for scalar - tensor
  proc f(x: T): T = x - val
  return t.map(f)

proc `.*`*[T: SomeNumber](a, b: Tensor[T]): Tensor[T] {.noSideEffect, inline.} =
  ## Element-wise multiplication (hadamard product)
  ## And broadcasted element-wise multiplication

  # FIXME: Can't use built-in proc `*` in map: https://github.com/nim-lang/Nim/issues/5702
  # map2(a, `+`, b)

  let (tmp_a, tmp_b) = unsafeBroadcast2(a, b)
  proc mul(x, y: T): T = x * y

  return map2(tmp_a, mul, tmp_b)

proc `./`*[T: SomeInteger](a, b: Tensor[T]): Tensor[T] {.noSideEffect, inline.} =
  ## Tensor element-wise division for integer numbers
  ## And broadcasted element-wise division
  let (tmp_a, tmp_b) = unsafeBroadcast2(a, b)
  proc dv(x, y: T): T = x div y
  return map2(tmp_a, dv, tmp_b)

proc `./`*[T: SomeReal](a, b: Tensor[T]): Tensor[T] {.noSideEffect, inline.} =
  ## Tensor element-wise division for real numbers
  ## And broadcasted element-wise division
  let (tmp_a, tmp_b) = unsafeBroadcast2(a, b)
  proc dv(x, y: T): T = x / y
  return map2(tmp_a, dv, tmp_b)

proc `./`*[T: SomeInteger](val: T, t: Tensor[T]): Tensor[T] {.noSideEffect, inline.} =
  ## Broadcasted division of an integer by a tensor of integer
  proc f(x: T): T = val div x
  return t.map(f)

proc `./`*[T: SomeReal](val: T, t: Tensor[T]): Tensor[T] {.noSideEffect, inline.} =
  ## Broadcasted division of a float by a tensor of floats
  proc f(x: T): T = val / x
  return t.map(f)