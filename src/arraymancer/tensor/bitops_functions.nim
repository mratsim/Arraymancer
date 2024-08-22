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
        ./ufunc
import std / bitops

export bitops

proc `shr`*[T1, T2: SomeInteger](t: Tensor[T1], value: T2): Tensor[T1] {.noinit.} =
  ## Broadcasted tensor-value `shr` (i.e. shift right) operator
  ##
  ## This is similar to numpy's `right_shift` and Matlab's `bitsra`
  ## (or `bitshift` with a positive shift value).
  t.map_inline(x shr value)

proc `shr`*[T1, T2: SomeInteger](value: T1, t: Tensor[T2]): Tensor[T2] {.noinit.} =
  ## Broadcasted value-tensor `shr` (i.e. shift right) operator
  ##
  ## This is similar to numpy's `right_shift` and Matlab's `bitsra`
  ## (or `bitshift` with a positive shift value).
  t.map_inline(value shr x)

proc `shr`*[T: SomeInteger](t1, t2: Tensor[T]): Tensor[T] {.noinit.} =
  ## Tensor element-wise `shr` (i.e. shift right) broadcasted operator
  ##
  ## This is similar to numpy's `right_shift` and Matlab's `bitsra`
  ## (or `bitshift` with a positive shift value).
  let (tmp1, tmp2) = broadcast2(t1, t2)
  result = map2_inline(tmp1, tmp2, x shr y)

proc `shl`*[T1, T2: SomeInteger](t: Tensor[T1], value: T2): Tensor[T1] {.noinit.} =
  ## Broadcasted tensor-value `shl` (i.e. shift left) operator
  ##
  ## This is similar to numpy's `left_shift` and Matlab's `bitsla`
  ## (or `bitshift` with a negative shift value).
  t.map_inline(x shl value)

proc `shl`*[T1, T2: SomeInteger](value: T1, t: Tensor[T2]): Tensor[T2] {.noinit.} =
  ## Broadcasted value-tensor `shl` (i.e. shift left) operator
  ##
  ## This is similar to numpy's `left_shift` and Matlab's `bitsla`
  ## (or `bitshift` with a negative shift value).
  t.map_inline(value shl x)

proc `shl`*[T: SomeInteger](t1, t2: Tensor[T]): Tensor[T] {.noinit.} =
  ## Tensor element-wise `shl` (i.e. shift left) broadcasted operator
  ##
  ## This is similar to numpy's `left_shift` and Matlab's `bitsla`
  ## (or `bitshift` with a negative shift value).
  let (tmp1, tmp2) = broadcast2(t1, t2)
  result = map2_inline(tmp1, tmp2, x shl y)

makeUniversal(bitnot,
  docSuffix="""Element-wise `bitnot` procedure

This is similar to numpy's `bitwise_not` and Matlab's `bitnot`.""")

proc bitand*[T](t: Tensor[T], value: T): Tensor[T] {.noinit.} =
  ## Broadcasted tensor-value `bitand` procedure
  ##
  ## This is similar to numpy's `bitwise_and` and Matlab's `bitand`.
  t.map_inline(bitand(x, value))

proc bitand*[T](value: T, t: Tensor[T]): Tensor[T] {.noinit.} =
  ## Broadcasted value-tensor `bitand` procedure
  ##
  ## This is similar to numpy's `bitwise_and` and Matlab's `bitand`.
  t.map_inline(bitand(value, x))

proc bitand*[T](t1, t2: Tensor[T]): Tensor[T] {.noinit.} =
  ## Tensor element-wise `bitand` procedure
  ##
  ## This is similar to numpy's `bitwise_and` and Matlab's `bitand`.
  let (tmp1, tmp2) = broadcast2(t1, t2)
  result = map2_inline(tmp1, tmp2, bitand(x, y))


proc bitor*[T](t: Tensor[T], value: T): Tensor[T] {.noinit.} =
  ## Broadcasted tensor-value `bitor` procedure
  ##
  ## This is similar to numpy's `bitwise_or` and Matlab's `bitor`.
  t.map_inline(bitor(x, value))

proc bitor*[T](value: T, t: Tensor[T]): Tensor[T] {.noinit.} =
  ## Broadcasted value-tensor `bitor` procedure
  ##
  ## This is similar to numpy's `bitwise_or` and Matlab's `bitor`.
  t.map_inline(bitor(value, x))

proc bitor*[T](t1, t2: Tensor[T]): Tensor[T] {.noinit.} =
  ## Tensor element-wise `bitor` procedure
  ##
  ## This is similar to numpy's `bitwise_or` and Matlab's `bitor`.
  let (tmp1, tmp2) = broadcast2(t1, t2)
  map2_inline(tmp1, tmp2, bitor(x, y))

proc bitxor*[T](t: Tensor[T], value: T): Tensor[T] {.noinit.} =
  ## Broadcasted tensor-value `bitxor` procedure
  ##
  ## This is similar to numpy's `bitwise_xor` and Matlab's `bitxor`.
  t.map_inline(bitxor(x, value))

proc bitxor*[T](value: T, t: Tensor[T]): Tensor[T] {.noinit.} =
  ## Broadcasted value-tensor `bitxor` procedure
  ##
  ## This is similar to numpy's `bitwise_xor` and Matlab's `bitxor`.
  t.map_inline(bitxor(value, x))

proc bitxor*[T](t1, t2: Tensor[T]): Tensor[T] {.noinit.} =
  ## Tensor element-wise `bitxor` procedure
  ##
  ## This is similar to numpy's `bitwise_xor` and Matlab's `bitxor`.
  let (tmp1, tmp2) = broadcast2(t1, t2)
  map2_inline(tmp1, tmp2, bitxor(x, y))

makeUniversal(reverseBits,
  docSuffix="Element-wise `reverseBits` procedure")
