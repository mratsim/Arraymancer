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
        ./math
import complex except Complex64, Complex32

# #########################################################
# # Broadcasting Tensor-Tensor
# # And element-wise multiplication (Hadamard) and division

proc `.+`*[T: SomeNumber|Complex[float32]|Complex[float64]](a, b: Tensor[T]): Tensor[T] {.noInit,inline.} =
  ## Broadcasted addition for tensors of incompatible but broadcastable shape.
  let (tmp_a, tmp_b) = broadcast2(a, b)
  result = tmp_a + tmp_b

proc `.-`*[T: SomeNumber|Complex[float32]|Complex[float64]](a, b: Tensor[T]): Tensor[T] {.noInit,inline.} =
  ## Broadcasted addition for tensors of incompatible but broadcastable shape.
  let (tmp_a, tmp_b) = broadcast2(a, b)
  result = tmp_a - tmp_b

proc `.*`*[T: SomeNumber|Complex[float32]|Complex[float64]](a, b: Tensor[T]): Tensor[T] {.noInit.} =
  ## Element-wise multiplication (Hadamard product).
  ##
  ## And broadcasted element-wise multiplication.

  let (tmp_a, tmp_b) = broadcast2(a, b)
  result = map2_inline(tmp_a, tmp_b, x * y)

proc `./`*[T: SomeInteger](a, b: Tensor[T]): Tensor[T] {.noInit.} =
  ## Tensor element-wise division for integer numbers.
  ##
  ## And broadcasted element-wise division.
  let (tmp_a, tmp_b) = broadcast2(a, b)
  result = map2_inline(tmp_a, tmp_b, x div y)

proc `./`*[T: SomeFloat|Complex[float32]|Complex[float64]](a, b: Tensor[T]): Tensor[T] {.noInit.} =
  ## Tensor element-wise division for real numbers.
  ##
  ## And broadcasted element-wise division.
  let (tmp_a, tmp_b) = broadcast2(a, b)
  result = map2_inline(tmp_a, tmp_b, x / y )

# ##############################################
# # Broadcasting in-place Tensor-Tensor

proc `.+=`*[T: SomeNumber|Complex[float32]|Complex[float64]](a: var Tensor[T], b: Tensor[T]) =
  ## Tensor broadcasted in-place addition.
  ##
  ## Only the right hand side tensor can be broadcasted.
  # shape check done in apply2 proc

  let tmp_b = b.broadcast(a.shape)
  apply2_inline(a, tmp_b, x + y)

proc `.-=`*[T: SomeNumber|Complex[float32]|Complex[float64]](a: var Tensor[T], b: Tensor[T]) =
  ## Tensor broadcasted in-place substraction.
  ##
  ## Only the right hand side tensor can be broadcasted.
  # shape check done in apply2 proc

  let tmp_b = b.broadcast(a.shape)
  apply2_inline(a, tmp_b, x - y)

proc `.*=`*[T: SomeNumber|Complex[float32]|Complex[float64]](a: var Tensor[T], b: Tensor[T]) =
  ## Tensor broadcasted in-place multiplication (Hadamard product)
  ##
  ## Only the right hand side tensor can be broadcasted
  # shape check done in apply2 proc

  let tmp_b = b.broadcast(a.shape)
  apply2_inline(a, tmp_b, x * y)

proc `./=`*[T: SomeInteger](a: var Tensor[T], b: Tensor[T]) =
  ## Tensor broadcasted in-place integer division.
  ##
  ## Only the right hand side tensor can be broadcasted.
  # shape check done in apply2 proc

  let tmp_b = b.broadcast(a.shape)
  apply2_inline(a, tmp_b, x div y)

proc `./=`*[T: SomeFloat|Complex[float32]|Complex[float64]](a: var Tensor[T], b: Tensor[T]) =
  ## Tensor broadcasted in-place float division.
  ##
  ## Only the right hand side tensor can be broadcasted.
  # shape check done in apply2 proc

  let tmp_b = b.broadcast(a.shape)
  apply2_inline(a, tmp_b, x / y)


# ##############################################
# # Broadcasting Tensor-Scalar and Scalar-Tensor

proc `.+`*[T: SomeNumber|Complex[float32]|Complex[float64]](val: T, t: Tensor[T]): Tensor[T] {.noInit.} =
  ## Broadcasted addition for tensor + scalar.
  result = t.map_inline(x + val)

proc `.+`*[T: SomeNumber|Complex[float32]|Complex[float64]](t: Tensor[T], val: T): Tensor[T] {.noInit.} =
  ## Broadcasted addition for scalar + tensor.
  result = t.map_inline(x + val)

proc `.-`*[T: SomeNumber|Complex[float32]|Complex[float64]](val: T, t: Tensor[T]): Tensor[T] {.noInit.} =
  ## Broadcasted substraction for tensor - scalar.
  result = t.map_inline(val - x)

proc `.-`*[T: SomeNumber|Complex[float32]|Complex[float64]](t: Tensor[T], val: T): Tensor[T] {.noInit.} =
  ## Broadcasted substraction for scalar - tensor.
  result = t.map_inline(x - val)

proc `./`*[T: SomeInteger](val: T, t: Tensor[T]): Tensor[T] {.noInit.} =
  ## Broadcasted division of an integer by a tensor of integers.
  result = t.map_inline(val div x)

proc `./`*[T: SomeFloat|Complex[float32]|Complex[float64]](val: T, t: Tensor[T]): Tensor[T] {.noInit.} =
  ## Broadcasted division of a float by a tensor of floats.
  result = t.map_inline(val / x)

proc `./`*[T: SomeInteger](t: Tensor[T], val: T): Tensor[T] {.noInit.} =
  ## Broadcasted division of tensor of integers by an integer.
  result = t.map_inline(x div val)

proc `./`*[T: SomeFloat|Complex[float32]|Complex[float64]](t: Tensor[T], val: T): Tensor[T] {.noInit.} =
  ## Broadcasted division of a tensor of floats by a float.
  result = t.map_inline(x / val)

proc `.^`*[T: SomeFloat|Complex[float32]|Complex[float64]](t: Tensor[T], exponent: T): Tensor[T] {.noInit.} =
  ## Compute element-wise exponentiation
  result = t.map_inline pow(x, exponent)

# #####################################
# # Broadcasting in-place Tensor-Scalar

proc `.+=`*[T: SomeNumber|Complex[float32]|Complex[float64]](t: var Tensor[T], val: T) =
  ## Tensor in-place addition with a broadcasted scalar.
  t.apply_inline(x + val)

proc `.-=`*[T: SomeNumber|Complex[float32]|Complex[float64]](t: var Tensor[T], val: T) =
  ## Tensor in-place substraction with a broadcasted scalar.
  t.apply_inline(x - val)

proc `.^=`*[T: SomeFloat|Complex[float32]|Complex[float64]](t: var Tensor[T], exponent: T) =
  ## Compute in-place element-wise exponentiation
  t.apply_inline pow(x, exponent)

proc `.*=`*[T: SomeNumber|Complex[float32]|Complex[float64]](t: var Tensor[T], val: T) =
  ## Tensor in-place multiplication with a broadcasted scalar.
  t.apply_inline(x * val)

proc `./=`*[T: SomeNumber|Complex[float32]|Complex[float64]](t: var Tensor[T], val: T) =
  ## Tensor in-place division with a broadcasted scalar.
  t.apply_inline(x - val)
