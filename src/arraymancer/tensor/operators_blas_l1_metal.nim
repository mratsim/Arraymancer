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

import  ./backend/metal/metal_backend,
        ./private/p_init_metal,
        ./private/p_checks_metal,
        ./data_structure

# ####################################################################
# BLAS Level 1 (Vector dot product, Addition, Scalar to Vector/Matrix)

proc dot*[T: SomeFloat](a, b: MetalTensor[T]): T {.inline.} =
  ## Vector to Vector dot (scalar) product using GPU reduction kernel
  when compileOption("boundChecks"):
    check_dot_prod(a, b)

  result = metalDot[T](a.storage.Fbuffer, b.storage.Fbuffer, a.storage.Flen)

proc `+=`*[T: SomeFloat](a: var MetalTensor[T], b: MetalTensor[T]) =
  ## MetalTensor in-place addition

  when compileOption("boundChecks"):
    check_elementwise(a, b)

  metalElementwise[T]("add",
    a.storage.Fbuffer,
    b.storage.Fbuffer,
    a.storage.Fbuffer,
    a.storage.Flen
  )

proc `+`*[T: SomeFloat](a, b: MetalTensor[T]): MetalTensor[T] {.noinit.} =
  ## MetalTensor addition

  when compileOption("boundChecks"):
    check_elementwise(a, b)

  result = newMetalTensor[T](a.shape)
  metalElementwise[T]("add",
    a.storage.Fbuffer,
    b.storage.Fbuffer,
    result.storage.Fbuffer,
    a.storage.Flen
  )

proc `-=`*[T: SomeFloat](a: var MetalTensor[T], b: MetalTensor[T]) =
  ## MetalTensor in-place substraction

  when compileOption("boundChecks"):
    check_elementwise(a, b)

  metalElementwise[T]("sub",
    a.storage.Fbuffer,
    b.storage.Fbuffer,
    a.storage.Fbuffer,
    a.storage.Flen
  )

proc `-`*[T: SomeFloat](a, b: MetalTensor[T]): MetalTensor[T] {.noinit.} =
  ## MetalTensor substraction

  when compileOption("boundChecks"):
    check_elementwise(a, b)

  result = newMetalTensor[T](a.shape)
  metalElementwise[T]("sub",
    a.storage.Fbuffer,
    b.storage.Fbuffer,
    result.storage.Fbuffer,
    a.storage.Flen
  )

proc `*=`*[T: SomeFloat](t: var MetalTensor[T]; a: T) {.inline.} =
  ## MetalTensor inplace multiplication by a scalar

  metalScalarMul[T](
    t.storage.Fbuffer,
    t.storage.Fbuffer,
    a,
    t.storage.Flen
  )

proc `*`*[T: SomeFloat](a: T, t: MetalTensor[T]): MetalTensor[T] {.noinit, inline.} =
  ## MetalTensor multiplication by a scalar

  result = newMetalTensor[T](t.shape)
  metalScalarMul[T](
    t.storage.Fbuffer,
    result.storage.Fbuffer,
    a,
    t.storage.Flen
  )
