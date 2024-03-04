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
        ./higher_order_applymap,
        ./shapeshifting,
        ../private/deprecate,
        ./private/p_empty_tensors

import std/math

# #########################################################
# # Broadcasting Tensor-Tensor
# # And element-wise multiplication (Hadamard) and division

proc `+.`*[T: SomeNumber|Complex[float32]|Complex[float64]](a, b: Tensor[T]): Tensor[T] {.noinit,inline.} =
  ## Broadcasted addition for tensors of incompatible but broadcastable shape.
  let (tmp_a, tmp_b) = broadcast2(a, b)
  result = tmp_a + tmp_b

proc `-.`*[T: SomeNumber|Complex[float32]|Complex[float64]](a, b: Tensor[T]): Tensor[T] {.noinit,inline.} =
  ## Broadcasted addition for tensors of incompatible but broadcastable shape.
  let (tmp_a, tmp_b) = broadcast2(a, b)
  result = tmp_a - tmp_b

proc `*.`*[T: SomeNumber|Complex[float32]|Complex[float64]](a, b: Tensor[T]): Tensor[T] {.noinit.} =
  ## Element-wise multiplication (Hadamard product).
  ##
  ## And broadcasted element-wise multiplication.
  let (tmp_a, tmp_b) = broadcast2(a, b)
  result = map2_inline(tmp_a, tmp_b, x * y)

proc `/.`*[T: SomeNumber|Complex[float32]|Complex[float64]](a, b: Tensor[T]): Tensor[T] {.noinit.} =
  ## Tensor element-wise division
  ##
  ## And broadcasted element-wise division.
  let (tmp_a, tmp_b) = broadcast2(a, b)
  when T is SomeInteger:
    result = map2_inline(tmp_a, tmp_b, x div y )
  else:
    result = map2_inline(tmp_a, tmp_b, x / y )

proc `mod`*[T: SomeNumber](a, b: Tensor[T]): Tensor[T] {.noinit.} =
  ## Tensor element-wise modulo operation
  ##
  ## And broadcasted element-wise modulo operation.
  let (tmp_a, tmp_b) = broadcast2(a, b)
  result = map2_inline(tmp_a, tmp_b, x mod y)

# ##############################################
# # Broadcasting in-place Tensor-Tensor

proc `+.=`*[T: SomeNumber|Complex[float32]|Complex[float64]](a: var Tensor[T], b: Tensor[T]) =
  ## Tensor broadcasted in-place addition.
  ##
  ## Only the right hand side tensor can be broadcasted.
  # shape check done in apply2 proc

  let tmp_b = b.broadcast(a.shape)
  apply2_inline(a, tmp_b, x + y)

proc `-.=`*[T: SomeNumber|Complex[float32]|Complex[float64]](a: var Tensor[T], b: Tensor[T]) =
  ## Tensor broadcasted in-place substraction.
  ##
  ## Only the right hand side tensor can be broadcasted.
  # shape check done in apply2 proc

  let tmp_b = b.broadcast(a.shape)
  apply2_inline(a, tmp_b, x - y)

proc `*.=`*[T: SomeNumber|Complex[float32]|Complex[float64]](a: var Tensor[T], b: Tensor[T]) =
  ## Tensor broadcasted in-place multiplication (Hadamard product)
  ##
  ## Only the right hand side tensor can be broadcasted
  # shape check done in apply2 proc

  let tmp_b = b.broadcast(a.shape)
  apply2_inline(a, tmp_b, x * y)

proc `/.=`*[T: SomeNumber|Complex[float32]|Complex[float64]](a: var Tensor[T], b: Tensor[T]) =
  ## Tensor broadcasted in-place division.
  ##
  ## Only the right hand side tensor can be broadcasted.
  # shape check done in apply2 proc

  let tmp_b = b.broadcast(a.shape)
  when T is SomeInteger:
    apply2_inline(a, tmp_b, x div y)
  else:
    apply2_inline(a, tmp_b, x / y)

# ##############################################
# # Broadcasting Tensor-Scalar and Scalar-Tensor

proc `+.`*[T: SomeNumber|Complex[float32]|Complex[float64]](val: T, t: Tensor[T]): Tensor[T] {.noinit.} =
  ## Broadcasted addition for tensor + scalar of the same type.
  returnEmptyIfEmpty(t)
  result = t.map_inline(x + val)

proc `+.`*[T: SomeNumber|Complex[float32]|Complex[float64]](t: Tensor[T], val: T): Tensor[T] {.noinit.} =
  ## Broadcasted addition for scalar + tensor of the same type.
  returnEmptyIfEmpty(t)
  result = t.map_inline(x + val)

proc `-.`*[T: SomeNumber|Complex[float32]|Complex[float64]](val: T, t: Tensor[T]): Tensor[T] {.noinit.} =
  ## Broadcasted substraction for tensor - scalar of the same type.
  returnEmptyIfEmpty(t)
  result = t.map_inline(val - x)

proc `-.`*[T: SomeNumber|Complex[float32]|Complex[float64]](t: Tensor[T], val: T): Tensor[T] {.noinit.} =
  ## Broadcasted substraction for scalar - tensor of the same type.
  returnEmptyIfEmpty(t)
  result = t.map_inline(x - val)

proc `*.`*[T: SomeNumber|Complex[float32]|Complex[float64]](val: T, t: Tensor[T]): Tensor[T] {.noinit.} =
  ## Broadcasted multiplication for tensor * scalar of the same type.
  returnEmptyIfEmpty(t)
  result = t.map_inline(x * val)

proc `*.`*[T: SomeNumber|Complex[float32]|Complex[float64]](t: Tensor[T], val: T): Tensor[T] {.noinit.} =
  ## Broadcasted multiplication for scalar * tensor of the same type.
  returnEmptyIfEmpty(t)
  result = t.map_inline(x * val)

proc `/.`*[T: SomeNumber|Complex[float32]|Complex[float64]](val: T, t: Tensor[T]): Tensor[T] {.noinit.} =
  ## Broadcasted division for scalar / tensor of the same type.
  returnEmptyIfEmpty(t)
  when T is SomeInteger:
    result = t.map_inline(val div x)
  else:
    result = t.map_inline(val / x)

proc `/.`*[T: SomeNumber|Complex[float32]|Complex[float64]](t: Tensor[T], val: T): Tensor[T] {.noinit.} =
  ## Broadcasted division for tensor / scalar of the same type.
  returnEmptyIfEmpty(t)
  when T is SomeInteger:
    result = t.map_inline(x div val)
  else:
    result = t.map_inline(x / val)

proc `^.`*[T: SomeFloat|Complex[float32]|Complex[float64]](t: Tensor[T], exponent: T): Tensor[T] {.noinit.} =
  ## Compute element-wise exponentiation: tensor ^ scalar of the same type.
  returnEmptyIfEmpty(t)
  result = t.map_inline pow(x, exponent)

proc `^.`*[T: SomeFloat|Complex[float32]|Complex[float64]](base: T, t: Tensor[T]): Tensor[T] {.noinit.} =
  ## Broadcasted exponentiation: scalar ^ tensor of the same type.
  returnEmptyIfEmpty(t)
  result = t.map_inline pow(base, x)

# #####################################
# # Broadcasting in-place Tensor-Scalar

proc `+.=`*[T: SomeNumber|Complex[float32]|Complex[float64]](t: var Tensor[T], val: T) =
  ## Tensor in-place addition with a broadcasted scalar.
  if t.size == 0:
    return
  t.apply_inline(x + val)

proc `-.=`*[T: SomeNumber|Complex[float32]|Complex[float64]](t: var Tensor[T], val: T) =
  ## Tensor in-place substraction with a broadcasted scalar.
  if t.size == 0:
    return
  t.apply_inline(x - val)

proc `^.=`*[T: SomeFloat|Complex[float32]|Complex[float64]](t: var Tensor[T], exponent: T) =
  ## Compute in-place element-wise exponentiation
  if t.size == 0:
    return
  t.apply_inline pow(x, exponent)

proc `*.=`*[T: SomeNumber|Complex[float32]|Complex[float64]](t: var Tensor[T], val: T) =
  ## Tensor in-place multiplication with a broadcasted scalar.
  if t.size == 0:
    return
  t.apply_inline(x * val)

proc `/.=`*[T: SomeNumber|Complex[float32]|Complex[float64]](t: var Tensor[T], val: T) =
  ## Tensor in-place division with a broadcasted scalar.
  if t.size == 0:
    return
  t.apply_inline(x / val)


# #################################################
# # Mixed Complex64 tensor - real scalar operations
#
# It is always possible to operate on a Complex64 tensor with a real scalar
# without loss of precission, so we allow it without explicit type conversion.
# This makes complex tensor arithmetic more convenient to use.

proc `+.`*[T: SomeNumber](val: T, t: Tensor[Complex64]): Tensor[Complex64] {.noinit, inline.} =
  ## Broadcasted addition for real scalar + Complex64 tensor
  ##
  ## The scalar is automatically converted to Complex64 before the operation.
  let complex_val = complex(float64(val))
  complex_val +. t

proc `+.`*[T: SomeNumber](t: Tensor[Complex64], val: T): Tensor[Complex64] {.noinit, inline.} =
  ## Broadcasted addition for real scalar + Complex64 tensor
  ##
  ## The scalar is automatically converted to Complex64 before the operation.
  let complex_val = complex(float64(val))
  t +. complex_val

proc `-.`*[T: SomeNumber](val: T, t: Tensor[Complex64]): Tensor[Complex64] {.noinit, inline.} =
  ## Broadcasted subtraction for real scalar - Complex64 tensor
  ##
  ## The scalar is automatically converted to Complex64 before the operation.
  let complex_val = complex(float64(val))
  complex_val -. t

proc `-.`*[T: SomeNumber](t: Tensor[Complex64], val: T): Tensor[Complex64] {.noinit, inline.} =
  ## Broadcasted subtraction for real scalar - Complex64 tensor
  ##
  ## The scalar is automatically converted to Complex64 before the operation.
  let complex_val = complex(float64(val))
  t -. complex_val

proc `*.`*[T: SomeNumber](val: T, t: Tensor[Complex64]): Tensor[Complex64] {.noinit, inline.} =
  ## Broadcasted multiplication for real scalar * Complex64 tensor
  ##
  ## The scalar is automatically converted to Complex64 before the operation.
  let complex_val = complex(float64(val))
  complex_val *. t

proc `*.`*[T: SomeNumber](t: Tensor[Complex64], val: T): Tensor[Complex64] {.noinit, inline.} =
  ## Broadcasted multiplication for real scalar * Complex64 tensor
  ##
  ## The scalar is automatically converted to Complex64 before the operation.
  let complex_val = complex(float64(val))
  t *. complex_val

proc `/.`*[T: SomeNumber](val: T, t: Tensor[Complex64]): Tensor[Complex64] {.noinit, inline.} =
  ## Broadcasted division for real scalar / Complex64 tensor
  ##
  ## The scalar is automatically converted to Complex64 before the operation.
  let complex_val = complex(float64(val))
  complex_val /. t

proc `/.`*[T: SomeNumber](t: Tensor[Complex64], val: T): Tensor[Complex64] {.noinit, inline.} =
  ## Broadcasted division for real scalar / Complex64 tensor
  ##
  ## The scalar is automatically converted to Complex64 before the operation.
  let complex_val = complex(float64(val))
  t /. complex_val

proc `^.`*[T: SomeNumber](val: T, t: Tensor[Complex64]): Tensor[Complex64] {.noinit, inline.} =
  ## Broadcasted exponentiation for real scalar ^ Complex64 tensor
  ##
  ## The scalar is automatically converted to Complex64 before the operation.
  let complex_val = complex(float64(val))
  complex_val ^. t

proc `^.`*[T: SomeNumber](t: Tensor[Complex64], val: T): Tensor[Complex64] {.noinit, inline.} =
  ## Broadcasted exponentiation for real scalar ^ Complex64 tensor
  ##
  ## The scalar is automatically converted to Complex64 before the operation.
  let complex_val = complex(float64(val))
  t ^. complex_val

proc `+.=`*[T: SomeNumber](t: var Tensor[Complex64], val: T) {.inline.} =
  ## Complex64 tensor in-place addition with a real scalar.
  ##
  ## The scalar is automatically converted to Complex64 before the operation.
  let complex_val = complex(float64(val))
  t +.= complex_val

proc `-.=`*[T: SomeNumber](t: var Tensor[Complex64], val: T) {.inline.} =
  ## Complex64 tensor in-place subtraction of a real scalar.
  ##
  ## The scalar is automatically converted to Complex64 before the operation.
  let complex_val = complex(float64(val))
  t -.= complex_val

proc `*.=`*[T: SomeNumber](t: var Tensor[Complex64], val: T) {.inline.} =
  ## Complex64 tensor in-place multiplication with a real scalar.
  ##
  ## The scalar is automatically converted to Complex64 before the operation.
  let complex_val = complex(float64(val))
  t *.= complex_val

proc `/.=`*[T: SomeNumber](t: var Tensor[Complex64], val: T) {.inline.} =
  ## Complex64 tensor in-place division by a real scalar.
  ##
  ## The scalar is automatically converted to Complex64 before the operation.
  let complex_val = complex(float64(val))
  t /.= complex_val

proc `^.=`*[T: SomeNumber](t: var Tensor[Complex64], val: T) {.inline.} =
  ## Complex64 tensor in-place exponentiation by a real scalar.
  ##
  ## The scalar is automatically converted to Complex64 before the operation.
  let complex_val = complex(float64(val))
  t ^.= complex_val


# ##############################################
# Deprecated syntax

implDeprecatedBy(`.+`, `+.`, exported = true)
implDeprecatedBy(`.-`, `-.`, exported = true)
implDeprecatedBy(`.*`, `*.`, exported = true)
implDeprecatedBy(`./`, `/.`, exported = true)
implDeprecatedBy(`.^`, `^.`, exported = true)

implDeprecatedBy(`.=+`, `+.=`, exported = true)
implDeprecatedBy(`.=-`, `-.=`, exported = true)
implDeprecatedBy(`.=*`, `*.=`, exported = true)
implDeprecatedBy(`.=/`, `/.=`, exported = true)
implDeprecatedBy(`.^=`, `^.=`, exported = true)
