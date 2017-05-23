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

proc check_nested_elements(shape: seq[int], len: int) {.noSideEffect.}=
  ## Compare the detected shape from flatten with the real length of the data
  ## Input:
  ##   -- A shape (sequence of int)
  ##   -- A length (int)
  if (shape.product != len):
    raise newException(IndexError, "Each nested sequence at the same level must have the same number of elements")

proc newTensor*(shape: seq[int], T: typedesc, B: static[Backend]): Tensor[B,T] {.noSideEffect.} =
  ## Creates a new Tensor
  ## Input:
  ##      - Shape of the Tensor
  ##      - Type of its elements
  ##      - Backend

  let strides = shape_to_strides(shape)

  result.shape = shape
  result.strides = strides
  result.data = newSeq[T](shape.product)
  result.offset = 0

proc toTensor*(s:openarray, B: static[Backend]): auto {.noSideEffect.} =
  ## Convert an openarray to a Tensor
  ## TODO: have Backend.Cpu as default. pending https://github.com/nim-lang/Nim/issues/5864
  let shape = s.shape
  let data = toSeq(flatIter(s))

  when compileOption("boundChecks"): check_nested_elements(shape, data.len)

  result = newTensor(shape, type(data[0]), B)
  result.data = data

## TODO add tests for zeros, ones and randomTensor
proc zeros*[T: SomeNumber](shape: seq[int], typ: typedesc[T], B: static[Backend]): Tensor[B,T] {.noSideEffect, inline.} =
  ## Creates a new Tensor filled with 0
  ## Input:
  ##      - Shape of the Tensor
  ##      - Type of its elements
  ##      - Backend
  return newTensor(shape, typ, B)

proc zeros_like*[B: static[Backend], T: SomeNumber](t: Tensor[B,T]): Tensor[B,T] {.noSideEffect, inline.} =
  ## Creates a new Tensor filled with 0 with the same shape as the input
  ## Input:
  ##      - Shape of the Tensor
  ##      - Type of its elements
  ##      - Backend
  return newTensor(t.shape, T, B)

proc ones*[T: SomeNumber](shape: seq[int], typ: typedesc[T], B: static[Backend]): Tensor[B,T] {.noSideEffect.} =
  ## Creates a new Tensor filled with 1
  ## Input:
  ##      - Shape of the Tensor
  ##      - Type of its elements
  ##      - Backend
  result = newTensor(shape, typ, B)
  result.data.applyIt(1)

proc ones_like*[B: static[Backend], T: SomeNumber](t: Tensor[B,T]): Tensor[B,T] {.noSideEffect.} =
  ## Creates a new Tensor filled with 0 with the same shape as the input
  ## and filled with 1
  ## Input:
  ##      - Tensor
  result = newTensor(t.shape, T, B)
  result.data.applyIt(1)

template randomTensorT[T](shape: seq[int], max_or_range: typed, seed: int): untyped =
  let strides = shape_to_strides(shape)

  result.shape = shape
  result.strides = strides
  result.offset = 0

  randomize(seed)
  result.data = newSeqWith(shape.product, random(max_or_range))

proc randomTensor*(shape: seq[int], max: float, seed: int, B: static[Backend]): Tensor[B,float] =
  ## Creates a new float Tensor filled with values between 0 and max
  randomTensorT[float](shape, max, seed)

proc randomTensor*(shape: seq[int], max: int, seed: int, B: static[Backend]): Tensor[B,int] =
  ## Creates a new int Tensor filled with values between 0 and max-1
  randomTensorT[int](shape, max, seed)

proc randomTensor*[T](shape: seq[int], slice: Slice[T], seed: int, B: static[Backend]): Tensor[B,T] =
  ## Creates a new int Tensor filled with values in the Slice range.
  randomTensorT[T](shape, slice, seed)