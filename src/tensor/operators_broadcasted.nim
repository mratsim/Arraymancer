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

import ./shapeshifting

# #########################################################
# # Broadcasting Tensor-Tensor
# # And element-wise multiplication (Hadamard) and division

proc `.+`*[T: SomeNumber](a, b: Tensor[T]): Tensor[T] {.inline.} =
  ## Broadcasted addition for tensors of incompatible but broadcastable shape.
  let (tmp_a, tmp_b) = unsafeBroadcast2(a, b)
  return tmp_a + tmp_b

proc `.-`*[T: SomeNumber](a, b: Tensor[T]): Tensor[T] {.inline.} =
  ## Broadcasted addition for tensors of incompatible but broadcastable shape.
  let (tmp_a, tmp_b) = unsafeBroadcast2(a, b)
  return tmp_a - tmp_b

proc `.*`*[T: SomeNumber](a, b: Tensor[T]): Tensor[T] {.inline.} =
  ## Element-wise multiplication (Hadamard product).
  ##
  ## And broadcasted element-wise multiplication.

  let (tmp_a, tmp_b) = unsafeBroadcast2(a, b)
  return map2T(tmp_a, tmp_b, x * y)

proc `./`*[T: SomeInteger](a, b: Tensor[T]): Tensor[T] {.inline.} =
  ## Tensor element-wise division for integer numbers.
  ##
  ## And broadcasted element-wise division.
  let (tmp_a, tmp_b) = unsafeBroadcast2(a, b)
  return map2T(tmp_a, tmp_b, x div y)

proc `./`*[T: SomeReal](a, b: Tensor[T]): Tensor[T] {.inline.} =
  ## Tensor element-wise division for real numbers.
  ##
  ## And broadcasted element-wise division.
  let (tmp_a, tmp_b) = unsafeBroadcast2(a, b)
  return map2T(tmp_a, tmp_b, x / y )

# ##############################################
# # Broadcasting in-place Tensor-Tensor

proc `.+=`*[T: SomeNumber](a: var Tensor[T], b: Tensor[T]) {.inline.} =
  ## Tensor broadcasted in-place addition.
  ##
  ## Only the right hand side tensor can be broadcasted.
  # shape check done in apply2 proc

  let tmp_b = b.unsafeBroadcast(a.shape)
  apply2T(a, tmp_b, x + y)

proc `.-=`*[T: SomeNumber](a: var Tensor[T], b: Tensor[T]) {.inline.} =
  ## Tensor broadcasted in-place substraction.
  ##
  ## Only the right hand side tensor can be broadcasted.
  # shape check done in apply2 proc

  let tmp_b = b.unsafeBroadcast(a.shape)
  apply2T(a, tmp_b, x - y)

proc `.*=`*[T: SomeNumber](a: var Tensor[T], b: Tensor[T]) {.inline.} =
  ## Tensor broadcasted in-place multiplication (Hadamard product)
  ##
  ## Only the right hand side tensor can be broadcasted
  # shape check done in apply2 proc

  let tmp_b = b.unsafeBroadcast(a.shape)
  apply2T(a, tmp_b, x * y)

proc `./=`*[T: SomeInteger](a: var Tensor[T], b: Tensor[T]) {.inline.} =
  ## Tensor broadcasted in-place integer division.
  ##
  ## Only the right hand side tensor can be broadcasted.
  # shape check done in apply2 proc

  let tmp_b = b.unsafeBroadcast(a.shape)
  apply2T(a, tmp_b, x div y)

proc `./=`*[T: SomeReal](a: var Tensor[T], b: Tensor[T]) {.inline.} =
  ## Tensor broadcasted in-place float division.
  ##
  ## Only the right hand side tensor can be broadcasted.
  # shape check done in apply2 proc

  let tmp_b = b.unsafeBroadcast(a.shape)
  apply2T(a, tmp_b, x / y)


# ##############################################
# # Broadcasting Tensor-Scalar and Scalar-Tensor

proc `.+`*[T: SomeNumber](val: T, t: Tensor[T]): Tensor[T] {.inline.} =
  ## Broadcasted addition for tensor + scalar.
  return t.mapT(x + val)

proc `.+`*[T: SomeNumber](t: Tensor[T], val: T): Tensor[T] {.inline.} =
  ## Broadcasted addition for scalar + tensor.
  return t.mapT(x + val)

proc `.-`*[T: SomeNumber](val: T, t: Tensor[T]): Tensor[T] {.inline.} =
  ## Broadcasted substraction for tensor - scalar.
  return t.mapT(val - x)

proc `.-`*[T: SomeNumber](t: Tensor[T], val: T): Tensor[T] {.inline.} =
  ## Broadcasted substraction for scalar - tensor.
  return t.mapT(x - val)

proc `./`*[T: SomeInteger](val: T, t: Tensor[T]): Tensor[T] {.inline.} =
  ## Broadcasted division of an integer by a tensor of integers.
  return t.mapT(val div x)

proc `./`*[T: SomeReal](val: T, t: Tensor[T]): Tensor[T] {.inline.} =
  ## Broadcasted division of a float by a tensor of floats.
  return t.mapT(val / x)

proc `.^`*[T: SomeReal](t: Tensor[T], exponent: T): Tensor[T] {.inline.} =
  ## Compute element-wise exponentiation
  return t.mapT pow(x, exponent)

# #####################################
# # Broadcasting in-place Tensor-Scalar

proc `.+=`*[T: SomeNumber](t: var Tensor[T], val: T) {.inline.} =
  ## Tensor in-place addition with a broadcasted scalar.
  t.applyT(x + val)

proc `.-=`*[T: SomeNumber](t: var Tensor[T], val: T) {.inline.} =
  ## Tensor in-place substraction with a broadcasted scalar.
  t.applyT(x - val)

proc `.^=`*[T: SomeReal](t: var Tensor[T], exponent: T) {.inline.} =
  ## Compute in-place element-wise exponentiation
  t.applyT pow(x, exponent)