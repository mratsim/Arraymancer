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

import  ./private/p_checks,
        ./private/p_empty_tensors,
        ./data_structure,
        ./accessors, ./higher_order_applymap,
        nimblas

# ####################################################################
# BLAS Level 1 (Vector dot product, Addition, Scalar to Vector/Matrix)

  # FIXME: Can't use built-in proc `+` in map: https://github.com/nim-lang/Nim/issues/5702
  # map2(a, `+`, b)

proc dot*[T: SomeFloat](a, b: Tensor[T]): T {.noSideEffect.} =
  ## Vector to Vector dot (scalar) product
  # TODO: blas's complex vector dot only support dotu and dotc
  when compileOption("boundChecks"): check_dot_prod(a,b)
  return dot(a.shape[0], a.get_offset_ptr, a.strides[0], b.get_offset_ptr, b.strides[0])

proc dot*[T: SomeInteger](a, b: Tensor[T]): T {.noSideEffect.} =
  ## Vector to Vector dot (scalar) product
  # Fallback for non-floats
  when compileOption("boundChecks"): check_dot_prod(a,b)
  for ai, bi in zip(a, b):
    result += ai * bi

# #########################################################
# # Tensor-Tensor linear algebra
# # shape checks are done in map2 proc

proc `+`*[T: SomeNumber|Complex[float32]|Complex[float64]](a, b: Tensor[T]): Tensor[T] {.noinit.} =
  ## Tensor addition
  map2_inline(a, b, x + y)

proc `-`*[T: SomeNumber|Complex[float32]|Complex[float64]](a, b: Tensor[T]): Tensor[T] {.noinit.} =
  ## Tensor substraction
  map2_inline(a, b, x - y)

# #########################################################
# # Tensor-Tensor in-place linear algebra

proc `+=`*[T: SomeNumber|Complex[float32]|Complex[float64]](a: var Tensor[T], b: Tensor[T]) =
  ## Tensor in-place addition
  a.apply2_inline(b, x + y)

proc `-=`*[T: SomeNumber|Complex[float32]|Complex[float64]](a: var Tensor[T], b: Tensor[T]) =
  ## Tensor in-place substraction
  a.apply2_inline(b, x - y)

# #########################################################
# # Tensor-scalar linear algebra

proc `*`*[T: SomeNumber|Complex[float32]|Complex[float64]](a: T, t: Tensor[T]): Tensor[T] {.noinit.} =
  ## Element-wise multiplication by a scalar
  returnEmptyIfEmpty(t)
  t.map_inline(x * a)

proc `*`*[T: SomeNumber|Complex[float32]|Complex[float64]](t: Tensor[T], a: T): Tensor[T] {.noinit.} =
  ## Element-wise multiplication by a scalar
  a * t

proc `/`*[T: SomeNumber|Complex[float32]|Complex[float64]](t: Tensor[T], a: T): Tensor[T] {.noinit.} =
  ## Element-wise division by a float scalar
  returnEmptyIfEmpty(t)
  when T is SomeInteger:
    t.map_inline(x div a)
  else:
    t.map_inline(x / a)

proc `div`*[T: SomeInteger](t: Tensor[T], a: T): Tensor[T] {.noinit.} =
  ## Element-wise division by an integer
  returnEmptyIfEmpty(t)
  t.map_inline(x div a)

proc `mod`*[T: SomeNumber](t: Tensor[T], val: T): Tensor[T] {.noinit.} =
  ## Broadcasted modulo operation
  returnEmptyIfEmpty(t)
  result = t.map_inline(x mod val)

proc `mod`*[T: SomeNumber](val: T, t: Tensor[T]): Tensor[T] {.noinit.} =
  ## Broadcasted modulo operation
  returnEmptyIfEmpty(t)
  result = t.map_inline(val mod x)

# Unsupported operations (these must be done using the broadcasted operators)
proc `+`*[T: SomeNumber|Complex[float32]|Complex[float64]](a: Tensor[T], val: T): Tensor[T] {.noinit,inline.} =
  ## Mathematical addition of tensors and scalars is undefined. Must use a broadcasted addition instead
  {.error: "To add a scalar to a tensor you must use the `+.` operator (instead of a plain `+` operator)".}

# Unsupported operations
proc `+`*[T: SomeNumber|Complex[float32]|Complex[float64]](val: T, a: Tensor[T]): Tensor[T] {.noinit,inline.} =
  ## Mathematical addition of tensors and scalars is undefined. Must use a broadcasted addition instead
  {.error: "To add a tensor to a scalar you must use the `+.` operator (instead of a plain `+` operator)".}

proc `-`*[T: SomeNumber|Complex[float32]|Complex[float64]](a: Tensor[T], val: T): Tensor[T] {.noinit,inline.} =
  ## Mathematical subtraction of tensors and scalars is undefined. Must use a broadcasted addition instead
  {.error: "To subtract a scalar from a tensor you must use the `-.` operator (instead of a plain `-` operator)".}

# Unsupported operations
proc `-`*[T: SomeNumber|Complex[float32]|Complex[float64]](val: T, a: Tensor[T]): Tensor[T] {.noinit,inline.} =
  ## Mathematical subtraction of tensors and scalars is undefined. Must use a broadcasted addition instead
  {.error: "To subtract a tensor from a scalar you must use the `-.` operator (instead of a plain `-` operator)".}

# #########################################################
# # Tensor-scalar in-place linear algebra

proc `*=`*[T: SomeNumber|Complex[float32]|Complex[float64]](t: var Tensor[T], a: T) =
  ## Element-wise multiplication by a scalar (in-place)
  if t.size == 0:
    return
  t.apply_inline(x * a)

proc `/=`*[T: SomeFloat|Complex[float32]|Complex[float64]](t: var Tensor[T], a: T) =
  ## Element-wise division by a scalar (in-place)
  if t.size == 0:
    return
  t.apply_inline(x / a)

proc `/=`*[T: SomeInteger](t: var Tensor[T], a: T) =
  ## Element-wise division by a scalar (in-place)
  if t.size == 0:
    return
  t.apply_inline(x div a)
