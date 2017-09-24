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

proc unsafeView*[T](t: Tensor[T]): Tensor[T] {.noSideEffect, inline.}=
  ## Input:
  ##     - A tensor
  ##       WARNING: even if the input tensor is a "let"
  ##       using this procedure does not guarantee immutability
  ## Returns:
  ##     - A shallow copy.
  ##
  ## WARNING !
  ##   Both tensors shares the same memory. Data modification on one will be reflected on the other.
  ##   However modifying the shape, strides or offset will not affect the other.
  result.shape = t.shape
  result.strides = t.strides
  result.offset = t.offset
  shallowCopy(result.data, t.data)

proc check_nested_elements(shape: seq[int], len: int) {.noSideEffect, inline.}=
  ## Compare the detected shape from flatten with the real length of the data
  ## Input:
  ##   -- A shape (sequence of int)
  ##   -- A length (int)
  if (shape.product != len):
    raise newException(IndexError, "Each nested sequence at the same level must have the same number of elements")

template tensorCpu[T](out_shape: openarray[int], t: Tensor[T], layout: OrderType = rowMajor): untyped =
  t.shape = @out_shape
  t.strides = shape_to_strides(t.shape, layout)
  t.offset = 0

template toTensorCpu(s: typed): untyped =
  let shape = s.shape
  let data = toSeq(flatIter(s))

  when compileOption("boundChecks"): check_nested_elements(shape, data.len)

  var t: Tensor[type(data[0])]
  tensorCpu(shape, t)
  t.data = data
  return t

proc newTensor*(shape: openarray[int], T: typedesc): Tensor[T] {.noSideEffect, inline.} =
  ## Creates a new Tensor on Cpu backend
  ## Input:
  ##      - Shape of the Tensor
  ##      - Type of its elements
  ## Result:
  ##      - A Tensor of the proper shape initialized with
  ##        the default type value (0 for numeric types) on Cpu backend
  tensorCpu(shape, result)
  result.data = newSeq[T](result.size)

proc toTensor*(s:openarray, dummy_bugfix: static[int] = 0 ): auto {.noSideEffect.} =
  ## Convert an openarray to a Tensor
  ## Nim >0.17 needed as "static[int] = 0" is not working in Nim 0.17
  ## Dummy_bugfix param is necessary due to: https://github.com/nim-lang/Nim/issues/6343
  # TODO: remove 'dummy_bugfix'
  toTensorCpu(s)

proc unsafeToTensor*[T: SomeNumber](data: seq[T]): Tensor[T] {.noSideEffect.} =
  ## Convert a seq to a Tensor, sharing the seq data
  ## Input:
  ##      - A seq with the tensor data
  ## Result:
  ##      - A rank 1 tensor with the same size of the input
  ## WARNING: result share storage with input
  tensorCpu([data.len], result)
  shallowCopy(result.data, data)

proc toTensor*(s:string): auto {.noSideEffect.} =
  ## Convert an openarray to a Tensor
  ##
  ## Handle string specifically (otherwise they are interpreted as openarray[char])
  toTensorCpu(s)

# TODO add tests for zeros, ones and randomTensor
proc zeros*[T: SomeNumber](shape: openarray[int], typ: typedesc[T]): Tensor[T] {.noSideEffect, inline.} =
  ## Creates a new Tensor filled with 0
  ##
  ## Input:
  ##      - Shape of the Tensor
  ##      - Type of its elements
  ## Result:
  ##      - A zero-ed Tensor of the input shape on backend Cpu
  return newTensor(shape, typ)

proc zeros_like*[T: SomeNumber](t: Tensor[T]): Tensor[T] {.noSideEffect, inline.} =
  ## Creates a new Tensor filled with 0 with the same shape as the input
  ## Input:
  ##      - Shape of the Tensor
  ##      - Type of its elements
  ## Result:
  ##      - A zero-ed Tensor of the same shape
  return zeros(t.shape, T)

proc ones*[T: SomeNumber](shape: openarray[int], typ: typedesc[T]): Tensor[T] {.noSideEffect,inline.} =
  ## Creates a new Tensor filled with 1
  ## Input:
  ##      - Shape of the Tensor
  ##      - Type of its elements
  ## Result:
  ##      - A one-ed Tensor of the same shape
  tensorCpu(shape, result)
  result.data = newSeqWith(result.size, 1.T)

proc ones_like*[T: SomeNumber](t: AnyTensor[T]): auto {.noSideEffect, inline.} =
  ## Creates a new Tensor filled with 0 with the same shape as the input
  ## and filled with 1
  ## Input:
  ##      - Tensor
  ## Result:
  ##      - A one-ed Tensor of the same shape
  return ones(t.shape, T)

template randomTensorCpu[T](t: Tensor[T], shape: openarray[int], max_or_range: typed): untyped =
  tensorCpu(shape, t)
  t.data = newSeqWith(t.size, random(max_or_range))

proc randomTensor*(shape: openarray[int], max: float): Tensor[float] =
  ## Creates a new float Tensor filled with values between 0 and max
  ## Random seed can be set by importing ``random`` and ``randomize(seed)``
  ## Input:
  ##      - a shape
  ##      - the max value possible (float)
  ##      - a tensor backend
  ## Result:
  ##      - A tensor of the input shape filled with random value between 0 and max input value
  randomTensorCpu(result, shape, max)

proc randomTensor*(shape: openarray[int], max: int): Tensor[int] =
  ## Creates a new int Tensor filled with values between 0 and max-1
  ## Random seed can be set by importing ``random`` and ``randomize(seed)``
  ## Input:
  ##      - a shape
  ##      - the max value possible (integer, exclusive)
  ##      - a tensor backend
  ## Result:
  ##      - A tensor of the input shape filled with random value between 0 and max input value (excluded)
  randomTensorCpu(result, shape, max)

proc randomTensor*[T](shape: openarray[int], slice: Slice[T]): Tensor[T] =
  ## Creates a new int Tensor filled with values in the Slice range.
  ## Random seed can be set by importing ``random`` and ``randomize(seed)``
  ## Input:
  ##      - a shape
  ##      - a range/slice
  ##      - a tensor backend
  ## Result:
  ##      - A tensor of the input shape filled with random value in the slice range
  randomTensorCpu(result, shape, slice)