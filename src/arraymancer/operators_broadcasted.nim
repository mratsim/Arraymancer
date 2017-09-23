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

# FIXME: Can't use built-in proc `+=` in map: https://github.com/nim-lang/Nim/issues/5702
# apply(a, `+=`, b)

##########################################################
## Broadcasting Tensor-Tensor
## And element-wise multiplication (Hadamard) and division

proc `.+`*[T: SomeNumber](a, b: Tensor[T]): Tensor[T] {.noSideEffect, inline.} =
  ## Broadcasted addition for tensors of incompatible but broadcastable shape
  let (tmp_a, tmp_b) = unsafeBroadcast2(a, b)
  return tmp_a + tmp_b

proc `.-`*[T: SomeNumber](a, b: Tensor[T]): Tensor[T] {.noSideEffect, inline.} =
  ## Broadcasted addition for tensors of incompatible but broadcastable shape
  let (tmp_a, tmp_b) = unsafeBroadcast2(a, b)
  return tmp_a - tmp_b

proc `.*`*[T: SomeNumber](a, b: Tensor[T]): Tensor[T] {.noSideEffect, inline.} =
  ## Element-wise multiplication (hadamard product)
  ## And broadcasted element-wise multiplication

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

###############################################
## Broadcasting in-place Tensor-Tensor

proc `.+=`*[T: SomeNumber](a: var Tensor[T], b: Tensor[T]) {.noSideEffect, inline.} =
  ## Tensor broadcasted in-place addition
  ## Only the right hand side tensor can be broadcaster
  # shape check done in apply2 proc

  let tmp_b = b.unsafeBroadcast(a.shape)
  proc inplace_add(x: var T, y: T) = x += y
  apply2(a, inplace_add, tmp_b)

proc `.-=`*[T: SomeNumber](a: var Tensor[T], b: Tensor[T]) {.noSideEffect, inline.} =
  ## Tensor broadcasted in-place substraction
  # shape check done in apply2 proc

  let tmp_b = b.unsafeBroadcast(a.shape)
  proc inplace_min(x: var T, y: T) = x -= y
  apply2(a, inplace_min, tmp_b)

proc `.*=`*[T: SomeNumber](a: var Tensor[T], b: Tensor[T]) {.noSideEffect, inline.} =
  ## Tensor broadcasted in-place element-wise multiplication
  # shape check done in apply2 proc

  let tmp_b = b.unsafeBroadcast(a.shape)
  proc inplace_mul(x: var T, y: T) = x *= y
  apply2(a, inplace_mul, tmp_b)

proc `./=`*[T: SomeInteger](a: var Tensor[T], b: Tensor[T]) {.noSideEffect, inline.} =
  ## Tensor broadcasted in-place element-wise multiplication
  # shape check done in apply2 proc

  let tmp_b = b.unsafeBroadcast(a.shape)
  proc inplace_div(x: var T, y: T) = x = x div y
  apply2(a, inplace_div, tmp_b)

proc `./=`*[T: SomeReal](a: var Tensor[T], b: Tensor[T]) {.noSideEffect, inline.} =
  ## Tensor broadcasted in-place element-wise multiplication
  # shape check done in apply2 proc

  let tmp_b = b.unsafeBroadcast(a.shape)
  proc inplace_div(x: var T, y: T) = x /= y
  apply2(a, inplace_div, tmp_b)


###############################################
## Broadcasting Tensor-Scalar and Scalar-Tensor

proc `.+`*[T: SomeNumber](val: T, t: Tensor[T]): Tensor[T] {.noSideEffect, inline.} =
  ## Broadcasted addition for tensor + scalar
  proc f(x: T): T = x + val
  return t.map(f)

proc `.+`*[T: SomeNumber](t: Tensor[T], val: T): Tensor[T] {.noSideEffect, inline.} =
  ## Broadcasted addition for scalar + tensor
  proc f(x: T): T = x + val
  return t.map(f)

proc `.-`*[T: SomeNumber](val: T, t: Tensor[T]): Tensor[T] {.noSideEffect, inline.} =
  ## Broadcasted substraction for tensor - scalar
  proc f(x: T): T = val - x
  return t.map(f)

proc `.-`*[T: SomeNumber](t: Tensor[T], val: T): Tensor[T] {.noSideEffect, inline.} =
  ## Broadcasted substraction for scalar - tensor
  proc f(x: T): T = x - val
  return t.map(f)

proc `./`*[T: SomeInteger](val: T, t: Tensor[T]): Tensor[T] {.noSideEffect, inline.} =
  ## Broadcasted division of an integer by a tensor of integer
  proc f(x: T): T = val div x
  return t.map(f)

proc `./`*[T: SomeReal](val: T, t: Tensor[T]): Tensor[T] {.noSideEffect, inline.} =
  ## Broadcasted division of a float by a tensor of floats
  proc f(x: T): T = val / x
  return t.map(f)

######################################
## Broadcasting in-place Tensor-Scalar

proc `.+=`*[T: SomeNumber](t: var Tensor[T], val: T) {.noSideEffect, inline.} =
  ## Tensor in-place addition with a broadcasted scalar

  proc f(x: var T) = x += val
  t.apply(f)

proc `.-=`*[T: SomeNumber](t: var Tensor[T], val: T) {.noSideEffect, inline.} =
  ## Tensor in-place substraction with a broadcasted scalar

  proc f(x: var T) = x -= val
  t.apply(f)