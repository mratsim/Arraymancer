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

# #########################################################
# # Tensor-Tensor linear algebra
# # shape checks are done in map2 proc

proc `+`*[T: SomeNumber](a, b: Tensor[T]): Tensor[T] =
  ## Tensor addition
  # Note: proc cannot be inlined for unknown reason (closure with env variable?)
  proc add_closure(x, y: T): T = x + y
  return map2(a, add_closure, b)

proc `-`*[T: SomeNumber](a, b: Tensor[T]): Tensor[T] =
  ## Tensor substraction
  # Note: proc cannot be inlined for unknown reason (closure with env variable?)
  proc sub_closure(x, y: T): T = x - y
  return map2(a, sub_closure, b)

# #########################################################
# # Tensor-Tensor in-place linear algebra

proc `+=`*[T: SomeNumber](a: var Tensor[T], b: Tensor[T]) {.inline.} =
  ## Tensor in-place addition
  # shape check done in apply2 proc

  proc madd_closure(x: var T, y: T) = x += y
  apply2(a, madd_closure, b)

proc `-=`*[T: SomeNumber](a: var Tensor[T], b: Tensor[T]) {.inline.} =
  ## Tensor in-place substraction
  proc msub_closure(x: var T, y: T) = x -= y
  apply2(a, msub_closure, b)

# #########################################################
# # Tensor-scalar linear algebra

proc `*`*[T: SomeNumber](a: T, t: Tensor[T]): Tensor[T] {.inline.} =
  ## Element-wise multiplication by a scalar
  proc scalmul_closure(x: T): T = a * x
  return t.map(scalmul_closure)

proc `*`*[T: SomeNumber](t: Tensor[T], a: T): Tensor[T] {.inline.} =
  ## Element-wise multiplication by a scalar
  a * t

proc `/`*[T: SomeReal](t: Tensor[T], a: T): Tensor[T] {.inline.} =
  ## Element-wise division by a float scalar
  proc scaldiv_closure(x: T): T = x / a
  return t.map(scaldiv_closure)

proc `div`*[T: SomeInteger](t: Tensor[T], a: T): Tensor[T] {.inline.} =
  ## Element-wise division by an integer
  proc scalintdiv_closure(x: T): T = x div a
  return t.map(scalintdiv_closure)

# #########################################################
# # Tensor-scalar in-place linear algebra

proc `*=`*[T: SomeNumber](t: var Tensor[T], a: T) {.inline.} =
  ## Element-wise multiplication by a scalar (in-place)
  proc mscalmul_closure(x: var T) = x *= a
  t.apply(mscalmul_closure)

proc `/=`*[T: SomeReal](t: var Tensor[T], a: T) {.inline.} =
  ## Element-wise division by a scalar (in-place)
  proc mscaldiv_closure(x: var T) = x /= a
  t.apply(mscaldiv_closure)

proc `/=`*[T: SomeInteger](t: var Tensor[T], a: T) {.inline.} =
  ## Element-wise division by a scalar (in-place)
  proc mscalintdiv_closure(x: T): T = x div a
  t.apply(mscalintdiv_closure)
