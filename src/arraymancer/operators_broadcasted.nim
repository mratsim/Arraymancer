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

# #########################################################
# # Broadcasting Tensor-Tensor
# # And element-wise multiplication (Hadamard) and division

proc `.+`*[T: SomeNumber](a, b: Tensor[T]): Tensor[T] {. inline.} =
  ## Broadcasted addition for tensors of incompatible but broadcastable shape.
  let (tmp_a, tmp_b) = unsafeBroadcast2(a, b)
  return tmp_a + tmp_b

proc `.-`*[T: SomeNumber](a, b: Tensor[T]): Tensor[T] {. inline.} =
  ## Broadcasted addition for tensors of incompatible but broadcastable shape.
  let (tmp_a, tmp_b) = unsafeBroadcast2(a, b)
  return tmp_a - tmp_b

proc `.*`*[T: SomeNumber](a, b: Tensor[T]): Tensor[T] {.inline.} =
  ## Element-wise multiplication (Hadamard product).
  ##
  ## And broadcasted element-wise multiplication.

  let (tmp_a, tmp_b) = unsafeBroadcast2(a, b)
  proc bc_mul_closure(x, y: T): T = x * y

  return map2(tmp_a, bc_mul_closure, tmp_b)

proc `./`*[T: SomeInteger](a, b: Tensor[T]): Tensor[T] {.inline.} =
  ## Tensor element-wise division for integer numbers.
  ##
  ## And broadcasted element-wise division.
  let (tmp_a, tmp_b) = unsafeBroadcast2(a, b)
  proc bc_intdiv_closure(x, y: T): T = x div y
  return map2(tmp_a, bc_intdiv_closure, tmp_b)

proc `./`*[T: SomeReal](a, b: Tensor[T]): Tensor[T] {.inline.} =
  ## Tensor element-wise division for real numbers.
  ##
  ## And broadcasted element-wise division.
  let (tmp_a, tmp_b) = unsafeBroadcast2(a, b)
  proc bc_div_closure(x, y: T): T = x / y
  return map2(tmp_a, bc_div_closure, tmp_b)

# ##############################################
# # Broadcasting in-place Tensor-Tensor

proc `.+=`*[T: SomeNumber](a: var Tensor[T], b: Tensor[T]) {. inline.} =
  ## Tensor broadcasted in-place addition.
  ##
  ## Only the right hand side tensor can be broadcasted.
  # shape check done in apply2 proc

  let tmp_b = b.unsafeBroadcast(a.shape)
  proc bc_madd_closure(x: var T, y: T) = x += y
  apply2(a, bc_madd_closure, tmp_b)

proc `.-=`*[T: SomeNumber](a: var Tensor[T], b: Tensor[T]) {. inline.} =
  ## Tensor broadcasted in-place substraction.
  ##
  ## Only the right hand side tensor can be broadcasted.
  # shape check done in apply2 proc

  let tmp_b = b.unsafeBroadcast(a.shape)
  proc bc_msub_closure(x: var T, y: T) = x -= y
  apply2(a, bc_msub_closure, tmp_b)

proc `.*=`*[T: SomeNumber](a: var Tensor[T], b: Tensor[T]) {. inline.} =
  ## Tensor broadcasted in-place multiplication (Hadamard product)
  ##
  ## Only the right hand side tensor can be broadcasted
  # shape check done in apply2 proc

  let tmp_b = b.unsafeBroadcast(a.shape)
  proc bc_mmul_closure(x: var T, y: T) = x *= y
  apply2(a, bc_mmul_closure, tmp_b)

proc `./=`*[T: SomeInteger](a: var Tensor[T], b: Tensor[T]) {. inline.} =
  ## Tensor broadcasted in-place integer division.
  ##
  ## Only the right hand side tensor can be broadcasted.
  # shape check done in apply2 proc

  let tmp_b = b.unsafeBroadcast(a.shape)
  proc bc_mintdiv_closure(x: var T, y: T) = x = x div y
  apply2(a, bc_mintdiv_closure, tmp_b)

proc `./=`*[T: SomeReal](a: var Tensor[T], b: Tensor[T]) {. inline.} =
  ## Tensor broadcasted in-place float division.
  ##
  ## Only the right hand side tensor can be broadcasted.
  # shape check done in apply2 proc

  let tmp_b = b.unsafeBroadcast(a.shape)
  proc bc_mdiv_closure(x: var T, y: T) = x /= y
  apply2(a, bc_mdiv_closure, tmp_b)


# ##############################################
# # Broadcasting Tensor-Scalar and Scalar-Tensor

proc `.+`*[T: SomeNumber](val: T, t: Tensor[T]): Tensor[T] {. inline.} =
  ## Broadcasted addition for tensor + scalar.
  proc bcs_add_closure(x: T): T = x + val
  return t.map(bcs_add_closure)

proc `.+`*[T: SomeNumber](t: Tensor[T], val: T): Tensor[T] {. inline.} =
  ## Broadcasted addition for scalar + tensor.
  proc bcs2_add_closure(x: T): T = x + val
  return t.map(bcs2_add_closure)

proc `.-`*[T: SomeNumber](val: T, t: Tensor[T]): Tensor[T] {. inline.} =
  ## Broadcasted substraction for tensor - scalar.
  proc bcs_min_closure(x: T): T = val - x
  return t.map(bcs_min_closure)

proc `.-`*[T: SomeNumber](t: Tensor[T], val: T): Tensor[T] {. inline.} =
  ## Broadcasted substraction for scalar - tensor.
  proc bcs2_min_closure(x: T): T = x - val
  return t.map(bcs2_min_closure)

proc `./`*[T: SomeInteger](val: T, t: Tensor[T]): Tensor[T] {. inline.} =
  ## Broadcasted division of an integer by a tensor of integers.
  proc bcs2_intdiv_closure(x: T): T = val div x
  return t.map(bcs2_intdiv_closure)

proc `./`*[T: SomeReal](val: T, t: Tensor[T]): Tensor[T] {. inline.} =
  ## Broadcasted division of a float by a tensor of floats.
  proc bcs2_div_closure(x: T): T = val / x
  return t.map(bcs2_div_closure)

proc `.^`*[T: SomeReal](t: Tensor[T], exponent: T): Tensor[T] {. inline.} =
  ## Compute element-wise exponentiation
  proc bc_pow_closure(x: T): T = pow(x, exponent)
  return t.map(bc_pow_closure)

# #####################################
# # Broadcasting in-place Tensor-Scalar

proc `.+=`*[T: SomeNumber](t: var Tensor[T], val: T) {. inline.} =
  ## Tensor in-place addition with a broadcasted scalar.

  proc bcs_madd_closure(x: var T) = x += val
  t.apply(bcs_madd_closure)

proc `.-=`*[T: SomeNumber](t: var Tensor[T], val: T) {. inline.} =
  ## Tensor in-place substraction with a broadcasted scalar.

  proc bcs_msub_closure(x: var T) = x -= val
  t.apply(bcs_msub_closure)

proc `.^=`*[T: SomeReal](t: var Tensor[T], exponent: T) {. inline.} =
  ## Compute in-place element-wise exponentiation

  proc bcs_mpow_closure(x: T): T = pow(x, exponent)
  t.apply(bcs_mpow_closure)