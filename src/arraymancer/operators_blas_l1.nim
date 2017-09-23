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


# Bounds checking functions
proc check_dot_prod(a, b:AnyTensor)  {.noSideEffect.}=
  if a.rank != 1 or b.rank != 1: raise newException(ValueError, "Dot product is only supported for vectors (tensors of rank 1)")
  if a.shape != b.shape: raise newException(ValueError, "Vector should be the same length")

# ####################################################################
# BLAS Level 1 (Vector dot product, Addition, Scalar to Vector/Matrix)

  # FIXME: Can't use built-in proc `+` in map: https://github.com/nim-lang/Nim/issues/5702
  # map2(a, `+`, b)

proc dot*[T: SomeReal](a, b: Tensor[T]): T {.noSideEffect, inline.} =
  ## Vector to Vector dot (scalar) product
  when compileOption("boundChecks"): check_dot_prod(a,b)
  return dot(a.shape[0], a.get_data_ptr, a.strides[0], b.get_data_ptr, b.strides[0])

proc dot*[T: SomeInteger](a, b: Tensor[T]): T {.noSideEffect.} =
  ## Vector to Vector dot (scalar) product
  # Fallback for non-floats
  when compileOption("boundChecks"): check_dot_prod(a,b)
  for ai, bi in zip(a.values, b.values):
    result += ai * bi

proc `+`*[T: SomeNumber](a, b: Tensor[T]): Tensor[T] {.noSideEffect.} =
  ## Tensor addition
  # shape check done in map2 proc

  # Note: proc cannot be inlined, probably due to the non in-place closure
  proc add(x, y: T): T = x + y
  return map2(a, add, b)

proc `+=`*[T: SomeNumber](a: var Tensor[T], b: Tensor[T]) {.noSideEffect, inline.} =
  ## Tensor in-place addition
  # shape check done in apply2 proc

  proc inplace_add(x: var T, y: T) = x += y
  apply2(a, inplace_add, b)

proc `-`*[T: SomeNumber](t: Tensor[T]): Tensor[T] {.noSideEffect, inline.} =
  ## Element-wise numerical negative
  t.map(proc(x: T): T = -x)

proc `-`*[T: SomeNumber](a, b: Tensor[T]): Tensor[T] {.noSideEffect.} =
  ## Tensor substraction
  # shape check done in map2 proc

  # Note: proc cannot be inlined, probably due to the non in-place closure
  proc sub(x, y: T): T = x - y
  return map2(a, sub, b)

proc `-=`*[T: SomeNumber](a: var Tensor[T], b: Tensor[T]) {.noSideEffect, inline.} =
  ## Tensor in-place addition
  # shape check done in apply2 proc

  proc inplace_min(x: var T, y: T) = x -= y
  apply2(a, inplace_min, b)

proc `*`*[T: SomeNumber](a: T, t: Tensor[T]): Tensor[T] {.noSideEffect, inline.} =
  ## Element-wise multiplication by a scalar
  proc f(x: T): T = a * x
  return t.map(f)

proc `*`*[T: SomeNumber](t: Tensor[T], a: T): Tensor[T] {.noSideEffect, inline.} =
  ## Element-wise multiplication by a scalar
  a * t

proc `*=`*[T: SomeNumber](t: var Tensor[T], a: T) {.noSideEffect, inline.} =
  ## Element-wise multiplication by a scalar (in-place)
  t.apply(proc(x: var T) = x *= a)

proc `/`*[T: SomeReal](t: Tensor[T], a: T): Tensor[T] {.noSideEffect, inline.} =
  ## Element-wise division by a float scalar
  proc f(x: T): T = x / a
  return t.map(f)

proc `div`*[T: SomeInteger](t: Tensor[T], a: T): Tensor[T] {.noSideEffect, inline.} =
  ## Element-wise division by an integer
  proc f(x: T): T = x / a
  return t.map(f)

proc `/=`*[T: SomeReal](t: var Tensor[T], a: T) {.noSideEffect, inline.} =
  ## Element-wise division by a scalar (in-place)
  t.apply(proc(x: var T) = x /= a)

proc `/=`*[T: SomeInteger](t: var Tensor[T], a: T) {.noSideEffect, inline.} =
  ## Element-wise division by a scalar (in-place)
  t.apply(proc(x: var T) = x = x div a)
