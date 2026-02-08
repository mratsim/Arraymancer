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
        ./data_structure,
        ./operators_blas_l1_metal,
        ./shapeshifting_metal

import ../private/deprecate

# #########################################################
# # Broadcasting Tensor-Tensor
# # And element-wise multiplication (Hadamard) and division

proc `+.`*[T: SomeFloat](a, b: MetalTensor[T]): MetalTensor[T] {.noinit,inline.} =
  ## Broadcasted addition for tensors of incompatible but broadcastable shape.
  let (tmp_a, tmp_b) = broadcast2(a, b)
  result = tmp_a + tmp_b

proc `-.`*[T: SomeFloat](a, b: MetalTensor[T]): MetalTensor[T] {.noinit,inline.} =
  ## Broadcasted subtraction for tensors of incompatible but broadcastable shape.
  let (tmp_a, tmp_b) = broadcast2(a, b)
  result = tmp_a - tmp_b

proc `*.`*[T: SomeFloat](a,b: MetalTensor[T]): MetalTensor[T] {.noinit.} =
  ## Element-wise multiplication (Hadamard product).
  ##
  ## And broadcasted element-wise multiplication.
  let (tmp_a, tmp_b) = broadcast2(a, b)
  
  result = newMetalTensor[T](tmp_a.shape)
  metalElementwise[T]("mul",
    tmp_a.storage.Fbuffer,
    tmp_b.storage.Fbuffer,
    result.storage.Fbuffer,
    result.storage.Flen
  )

proc `/.`*[T: SomeFloat](a, b: MetalTensor[T]): MetalTensor[T] {.noinit.} =
  ## Element-wise division.
  ##
  ## And broadcasted element-wise division.
  let (tmp_a, tmp_b) = broadcast2(a, b)
  
  result = newMetalTensor[T](tmp_a.shape)
  metalElementwise[T]("div",
    tmp_a.storage.Fbuffer,
    tmp_b.storage.Fbuffer,
    result.storage.Fbuffer,
    result.storage.Flen
  )

# ##############################################
# # Broadcasting in-place Tensor-Tensor

proc `+.=`*[T: SomeFloat](a: var MetalTensor[T], b: MetalTensor[T]) =
  ## Tensor broadcasted in-place addition.
  ##
  ## Only the right hand side tensor can be broadcasted.
  let tmp_b = b.broadcast(a.shape)
  a += tmp_b

proc `-.=`*[T: SomeFloat](a: var MetalTensor[T], b: MetalTensor[T]) =
  ## Tensor broadcasted in-place subtraction.
  ##
  ## Only the right hand side tensor can be broadcasted.
  let tmp_b = b.broadcast(a.shape)
  a -= tmp_b

proc `*.=`*[T: SomeFloat](a: var MetalTensor[T], b: MetalTensor[T]) =
  ## Tensor broadcasted in-place multiplication (Hadamard product)
  ##
  ## Only the right hand side tensor can be broadcasted.
  let tmp_b = b.broadcast(a.shape)
  metalElementwise[T]("mul",
    a.storage.Fbuffer,
    tmp_b.storage.Fbuffer,
    a.storage.Fbuffer,
    a.storage.Flen
  )

proc `/.=`*[T: SomeFloat](a: var MetalTensor[T], b: MetalTensor[T]) =
  ## Tensor broadcasted in-place float division.
  ##
  ## Only the right hand side tensor can be broadcasted.
  let tmp_b = b.broadcast(a.shape)
  metalElementwise[T]("div",
    a.storage.Fbuffer,
    tmp_b.storage.Fbuffer,
    a.storage.Fbuffer,
    a.storage.Flen
  )

# ##############################################
# # Broadcasting Tensor-Scalar and Scalar-Tensor

proc `+.`*[T: SomeFloat](t: MetalTensor[T], val: T): MetalTensor[T] {.noinit.} =
  ## Broadcasted addition for tensor + scalar.
  ## Uses GPU scalar_add kernel
  result = newMetalTensor[T](t.shape)
  metalScalarAdd[T](
    t.storage.Fbuffer,
    result.storage.Fbuffer,
    val,
    t.storage.Flen
  )

proc `-.`*[T: SomeFloat](t: MetalTensor[T], val: T): MetalTensor[T] {.noinit.} =
  ## Broadcasted subtraction for tensor - scalar.
  ## Implemented as tensor + (-scalar)
  result = newMetalTensor[T](t.shape)
  metalScalarAdd[T](
    t.storage.Fbuffer,
    result.storage.Fbuffer,
    -val,
    t.storage.Flen
  )

proc `+.`*[T: SomeFloat](val: T, t: MetalTensor[T]): MetalTensor[T] {.noinit.} =
  ## Broadcasted addition for scalar + tensor.
  result = t +. val

proc `-.`*[T: SomeFloat](val: T, t: MetalTensor[T]): MetalTensor[T] {.noinit.} =
  ## Broadcasted subtraction for scalar - tensor.
  ## Implemented as -(tensor - scalar)
  result = newMetalTensor[T](t.shape)
  metalScalarAdd[T](
    t.storage.Fbuffer,
    result.storage.Fbuffer,
    -val,
    t.storage.Flen
  )
  # Negate the result
  metalScalarMul[T](
    result.storage.Fbuffer,
    result.storage.Fbuffer,
    T(-1),
    t.storage.Flen
  )

proc `/.`*[T: SomeFloat](val: T, t: MetalTensor[T]): MetalTensor[T] {.noinit.} =
  ## Broadcasted division of a float by a tensor of floats.
  ## For now, fall back to CPU
  let cpuT = t.toCpu()
  let cpuResult = val /. cpuT
  result = cpuResult.metal()

# ##############################################
# # Broadcasting in-place Tensor-Scalar

proc `+.=`*[T: SomeFloat](t: var MetalTensor[T], val: T) =
  ## Broadcasted in-place addition for tensor += scalar.
  ## Uses GPU scalar_add kernel in-place
  metalScalarAdd[T](
    t.storage.Fbuffer,
    t.storage.Fbuffer,
    val,
    t.storage.Flen
  )

proc `-.=`*[T: SomeFloat](t: var MetalTensor[T], val: T) =
  ## Broadcasted in-place subtraction for tensor -= scalar.
  ## Implemented as tensor += (-scalar)
  metalScalarAdd[T](
    t.storage.Fbuffer,
    t.storage.Fbuffer,
    -val,
    t.storage.Flen
  )

# ##############################################
# Deprecated syntax

implDeprecatedBy(`.+`, `+.`, exported = true)
implDeprecatedBy(`.-`, `-.`, exported = true)
implDeprecatedBy(`.*`, `*.`, exported = true)
implDeprecatedBy(`./`, `/.`, exported = true)

implDeprecatedBy(`.=+`, `+.=`, exported = true)
implDeprecatedBy(`.=-`, `-.=`, exported = true)
implDeprecatedBy(`.=*`, `*.=`, exported = true)
implDeprecatedBy(`.=/`, `/.=`, exported = true)
