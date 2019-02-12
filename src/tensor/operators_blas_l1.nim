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
        ./data_structure,
        ./accessors, ./higher_order_applymap,
        nimblas
import complex except Complex32, Complex64

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

proc `+`*[T: SomeNumber|Complex[float32]|Complex[float64]](a, b: Tensor[T]): Tensor[T] {.noInit.} =
  ## Tensor addition
  map2_inline(a, b, x + y)

proc `-`*[T: SomeNumber|Complex[float32]|Complex[float64]](a, b: Tensor[T]): Tensor[T] {.noInit.} =
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

proc `*`*[T: SomeNumber|Complex[float32]|Complex[float64]](a: T, t: Tensor[T]): Tensor[T] {.noInit.} =
  ## Element-wise multiplication by a scalar
  t.map_inline(x * a)

proc `*`*[T: SomeNumber|Complex[float32]|Complex[float64]](t: Tensor[T], a: T): Tensor[T] {.noInit.} =
  ## Element-wise multiplication by a scalar
  a * t

proc `/`*[T: SomeFloat|Complex[float32]|Complex[float64]](t: Tensor[T], a: T): Tensor[T] {.noInit.} =
  ## Element-wise division by a float scalar
  t.map_inline(x / a)

proc `div`*[T: SomeInteger](t: Tensor[T], a: T): Tensor[T] {.noInit.} =
  ## Element-wise division by an integer
  t.map_inline(x div a)

# #########################################################
# # Tensor-scalar in-place linear algebra

proc `*=`*[T: SomeNumber|Complex[float32]|Complex[float64]](t: var Tensor[T], a: T) =
  ## Element-wise multiplication by a scalar (in-place)
  t.apply_inline(x * a)

proc `/=`*[T: SomeFloat|Complex[float32]|Complex[float64]](t: var Tensor[T], a: T) =
  ## Element-wise division by a scalar (in-place)
  t.apply_inline(x / a)

proc `/=`*[T: SomeInteger](t: var Tensor[T], a: T) =
  ## Element-wise division by a scalar (in-place)
  t.apply_inline(x div a)
