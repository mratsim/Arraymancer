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


template tensor[B,T](shape: openarray[int], result: Tensor[B,T]): untyped =
  result.shape = @shape
  result.strides = shape_to_strides(result.shape)
  result.offset = 0

proc newTensor*(shape: openarray[int], T: typedesc, B: static[Backend]): Tensor[B,T] {.noSideEffect.} =
  ## Creates a new Tensor
  ## Input:
  ##      - Shape of the Tensor
  ##      - Type of its elements
  ##      - Backend
  ## Result:
  ##      - A Tensor of the proper shape initialized with
  ##        the default type value (0 for numeric types)

  tensor(shape, result)
  result.data = newSeq[T](result.shape.product)

proc emptyTensor(shape: seq[int], T: typedesc, B: static[Backend]): Tensor[B,T] {.noSideEffect, inline.} =
  ## Creates an empty Tensor
  ## Internal proc so that toTensor has the proper internal type and backend
  tensor(shape, result)

template toTensorT(s: typed, B: static[Backend]): untyped =
  let shape = s.shape
  let data = toSeq(flatIter(s))

  when compileOption("boundChecks"): check_nested_elements(shape, data.len)

  result = emptyTensor(shape, type(data[0]), B)
  result.data = data

proc toTensor*(s:openarray, B: static[Backend]): auto {.noSideEffect.} =
  ## Convert an openarray to a Tensor
  # TODO: have Backend.Cpu as default. pending https://github.com/nim-lang/Nim/issues/5864
  toTensorT(s,B)

proc toTensor*(s:string, B: static[Backend]): auto {.noSideEffect.} =
  ## Convert an openarray to a Tensor
  ##
  ## Handle string specifically (otherwise they are interpreted as openarray[char])
  toTensorT(s,B)

# TODO add tests for zeros, ones and randomTensor
proc zeros*[T: SomeNumber](shape: openarray[int], typ: typedesc[T], B: static[Backend]): Tensor[B,T] {.noSideEffect, inline.} =
  ## Creates a new Tensor filled with 0
  ## Input:
  ##      - Shape of the Tensor
  ##      - Type of its elements
  ##      - Backend
  ## Result:
  ##      - A zero-ed Tensor of the input shape
  return newTensor(shape, typ, B)

proc zeros_like*[B: static[Backend], T: SomeNumber](t: Tensor[B,T]): Tensor[B,T] {.noSideEffect, inline.} =
  ## Creates a new Tensor filled with 0 with the same shape as the input
  ## Input:
  ##      - Shape of the Tensor
  ##      - Type of its elements
  ##      - Backend
  ## Result:
  ##      - A zero-ed Tensor of the same shape
  return zeros(t.shape, T, B)

proc ones*[T: SomeNumber](shape: openarray[int], typ: typedesc[T], B: static[Backend]): Tensor[B,T] {.noSideEffect.} =
  ## Creates a new Tensor filled with 1
  ## Input:
  ##      - Shape of the Tensor
  ##      - Type of its elements
  ##      - Backend
  ## Result:
  ##      - A one-ed Tensor of the same shape
  tensor(shape, result)
  result.data = newSeqWith(result.shape.product, 1.T)

proc ones_like*[B: static[Backend], T: SomeNumber](t: Tensor[B,T]): Tensor[B,T] {.noSideEffect, inline.} =
  ## Creates a new Tensor filled with 0 with the same shape as the input
  ## and filled with 1
  ## Input:
  ##      - Tensor
  ## Result:
  ##      - A one-ed Tensor of the same shape
  return ones(t.shape, T, B)

template randomTensorT(shape: openarray[int], max_or_range: typed): untyped =
  tensor(shape, result)
  result.data = newSeqWith(result.shape.product, random(max_or_range))

proc randomTensor*(shape: openarray[int], max: float, B: static[Backend]): Tensor[B,float] =
  ## Creates a new float Tensor filled with values between 0 and max
  ## Random seed can be set by importing ``random`` and ``randomize(seed)``
  ## Input:
  ##      - a shape
  ##      - the max value possible (float)
  ##      - a tensor backend
  ## Result:
  ##      - A tensor of the input shape filled with random value between 0 and max input value
  randomTensorT(shape, max)

proc randomTensor*(shape: openarray[int], max: int, B: static[Backend]): Tensor[B,int] =
  ## Creates a new int Tensor filled with values between 0 and max-1
  ## Random seed can be set by importing ``random`` and ``randomize(seed)``
  ## Input:
  ##      - a shape
  ##      - the max value possible (integer, exclusive)
  ##      - a tensor backend
  ## Result:
  ##      - A tensor of the input shape filled with random value between 0 and max input value (excluded)
  randomTensorT(shape, max)

proc randomTensor*[T](shape: openarray[int], slice: Slice[T], B: static[Backend]): Tensor[B,T] =
  ## Creates a new int Tensor filled with values in the Slice range.
  ## Random seed can be set by importing ``random`` and ``randomize(seed)``
  ## Input:
  ##      - a shape
  ##      - a range/slice
  ##      - a tensor backend
  ## Result:
  ##      - A tensor of the input shape filled with random value in the slice range
  randomTensorT(shape, slice)