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

proc `.*`*[T: SomeReal](a, b: Tensor[T]): T {.noSideEffect.} =
  ## Vector to Vector dot (scalar) product
  when compileOption("boundChecks"): check_dot_prod(a,b)
  return dot(a.shape[0], a.get_data_ptr, a.strides[0], b.get_data_ptr, b.strides[0])

proc `.*`*[T: SomeInteger](a, b: Tensor[T]): T {.noSideEffect.} =
  ## Vector to Vector dot (scalar) product
  # Fallback for non-floats
  when compileOption("boundChecks"): check_dot_prod(a,b)
  for ai, bi in zip(a.values, b.values):
    result += ai * bi

proc `+`*[T: SomeNumber](a, b: Tensor[T]): Tensor[T] {.noSideEffect.} =
  ## Tensor addition
  when compileOption("boundChecks"): check_elementwise(a,b)

  result.shape = a.shape
  result.strides = shape_to_strides(a.shape)
  result.data = newSeq[T](a.shape.product)
  result.offset = 0

  ## TODO use mitems instead of result.data[i] cf profiling
  for i, ai, bi in enumerate_zip(a.values, b.values):
    result.data[i] = ai + bi

proc `+=`*[T: SomeNumber](a: var Tensor[T], b: Tensor[T]) {.noSideEffect.} =
  ## Tensor in-place addition
  when compileOption("boundChecks"): check_elementwise(a,b)

  ## TODO: yield mutable values for a: https://forum.nim-lang.org/t/2972
  for a_idx, b_val in zip(a.real_indices, b.values):
    a.data[a_idx] += b_val

proc `-`*[T: SomeNumber](t: Tensor[T]): Tensor[T] {.noSideEffect.} =
  ## Element-wise numerical negative
  return t.fmap(proc(x: T): T = -x)

proc `-`*[T: SomeNumber](a, b: Tensor[T]): Tensor[T] {.noSideEffect.} =
  ## Tensor addition
  when compileOption("boundChecks"): check_elementwise(a,b)

  result.shape = a.shape
  result.strides = shape_to_strides(result.shape)
  result.data = newSeq[T](result.shape.product)
  result.offset = 0

  # TODO use mitems instead of result.data[i] cf profiling
  for i, ai, bi in enumerate_zip(a.values, b.values):
    result.data[i] = ai - bi

proc `-=`*[T: SomeNumber](a: var Tensor[T], b: Tensor[T]) {.noSideEffect.} =
  ## Tensor in-place addition
  when compileOption("boundChecks"): check_elementwise(a,b)

  # TODO: yield mutable values for a: https://forum.nim-lang.org/t/2972
  for a_idx, b_val in zip(a.real_indices, b.values):
    a.data[a_idx] -= b_val

proc `*`*[T: SomeNumber](a: T, t: Tensor[T]): Tensor[T] {.noSideEffect.} =
  ## Element-wise multiplication by a scalar
  proc f(x: T): T = a * x
  return t.fmap(f)

proc `*`*[T: SomeNumber](t: Tensor[T], a: T): Tensor[T] {.noSideEffect, inline.} =
  ## Element-wise multiplication by a scalar
  a * t

proc `*=`*[T: SomeNumber](t: var Tensor[T], a: T) {.noSideEffect.} =
  ## Element-wise multiplication by a scalar (in-place)
  t.apply(proc(x: T): T = a * x)

proc `/`*[T: SomeNumber](t: Tensor[T], a: T): Tensor[T] {.noSideEffect.} =
  ## Element-wise division by a scalar
  proc f(x: T): T = x / a
  return t.fmap(f)

proc `/=`*[T: SomeNumber](t: var Tensor[T], a: T): Tensor[T] {.noSideEffect.} =
  ## Element-wise division by a scalar (in-place)
  t.apply(proc(x: T): T = a / x)
