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

import  ../private/[functional, nested_containers, sequninit],
        ./backend/metadataArray,
        ./private/p_checks,
        ./private/p_init_cpu,
        ./data_structure,
        nimblas,
        sequtils,
        random,
        math

proc unsafeView*[T](t: Tensor[T]): Tensor[T] {.noSideEffect,noInit,inline, deprecated.}=
  ## Input:
  ##     - A tensor
  ## Returns:
  ##     - A shallow copy. Both tensors share the same memory location.
  ##
  ## Warning ⚠
  ##   Both tensors shares the same memory. Data modification on one will be reflected on the other.
  ##   However modifying the shape, strides or offset will not affect the other.
  result.shape = t.shape
  result.strides = t.strides
  result.offset = t.offset
  shallowCopy(result.data, t.data)

proc newTensorUninit*[T](shape: varargs[int]): Tensor[T] {.noSideEffect,noInit, inline.} =
  ## Creates a new Tensor on Cpu backend
  ## Input:
  ##      - Shape of the Tensor
  ##      - Type of its elements
  ## Result:
  ##      - A Tensor of the proper shape with NO initialization
  ## Warning ⚠
  ##   Tensor data is uninitialized an contains garbage.
  tensorCpu(shape, result)
  result.data = newSeqUninit[T](result.size)

proc newTensorUninit*[T](shape: MetadataArray): Tensor[T] {.noSideEffect,noInit, inline.} =
  ## Creates a new Tensor on Cpu backend
  ## Input:
  ##      - Shape of the Tensor
  ##      - Type of its elements
  ## Result:
  ##      - A Tensor of the proper shape with NO initialization
  ## Warning ⚠
  ##   Tensor data is uninitialized an contains garbage.
  tensorCpu(shape, result)
  result.data = newSeqUninit[T](result.size)

proc newTensor*[T](shape: varargs[int]): Tensor[T] {.noSideEffect,noInit, inline.} =
  ## Creates a new Tensor on Cpu backend
  ## Input:
  ##      - Shape of the Tensor
  ##      - Type of its elements
  ## Result:
  ##      - A Tensor of the proper shape initialized with
  ##        the default type value (0 for numeric types) on Cpu backend
  tensorCpu(shape, result)
  result.data = newSeq[T](result.size)

proc newTensorWith*[T](shape: varargs[int], value: T): Tensor[T] {.noSideEffect,noInit, inline.} =
  ## Creates a new Tensor filled with the given value
  ## Input:
  ##      - Shape of the Tensor
  ##      - Type of its elements
  ##      - Value to initialize its elements
  ## Result:
  ##      - A Tensor of the proper shape initialized with
  ##        the given value
  # Todo: use a template that can accept proc or value. See the code for newSeqWith: https://github.com/nim-lang/Nim/blob/master/lib/pure/collections/sequtils.nim#L650-L665
  tensorCpu(shape, result)
  result.data = newSeqWith(result.size, value)

proc toTensor*(s:openarray, dummy_bugfix: static[int] = 0 ): auto {.noSideEffect.} =
  ## Convert an openarray to a Tensor
  ## Input:
  ##      - An array or a seq (can be nested)
  ## Result:
  ##      - A Tensor of the same shape
  ##
  ## Note: dummy_bugfix param is unused and is a workaround a Nim bug.
  # TODO: remove 'dummy_bugfix' - https://github.com/nim-lang/Nim/issues/6343
  toTensorCpu(s)

proc unsafeToTensor*[T: SomeNumber](data: seq[T]): Tensor[T] {.noInit,noSideEffect.} =
  ## Convert a seq to a Tensor, sharing the seq data
  ## Input:
  ##      - A seq with the tensor data
  ## Result:
  ##      - A rank 1 tensor with the same size of the input
  ## WARNING: result share storage with input
  tensorCpu([data.len], result)
  shallowCopy(result.data, data)

proc toTensor*(s:string): auto {.noSideEffect.} =
  ## Convert a string to a Tensor
  ##
  ## This proc handles string specifically as otherwise they are interpreted as a sequence of char
  toTensorCpu(s)

proc zeros*[T: SomeNumber](shape: varargs[int]): Tensor[T] {.noInit,noSideEffect, inline.} =
  ## Creates a new Tensor filled with 0
  ##
  ## Input:
  ##      - Shape of the Tensor
  ##      - Type of its elements
  ## Result:
  ##      - A zero-ed Tensor of the input shape on backend Cpu
  tensorCpu(shape, result)
  result.data = newSeq[T](result.size)

proc zeros*[T: SomeNumber](shape: MetadataArray): Tensor[T] {.noInit,noSideEffect, inline.} =
  ## Creates a new Tensor filled with 0
  ##
  ## Input:
  ##      - Shape of the Tensor
  ##      - Type of its elements
  ## Result:
  ##      - A zero-ed Tensor of the input shape on backend Cpu
  tensorCpu(shape, result)
  result.data = newSeq[T](result.size)

proc zeros_like*[T: SomeNumber](t: Tensor[T]): Tensor[T] {.noInit,noSideEffect, inline.} =
  ## Creates a new Tensor filled with 0 with the same shape as the input
  ## Input:
  ##      - Shape of the Tensor
  ##      - Type of its elements
  ## Result:
  ##      - A zero-ed Tensor of the same shape
  return zeros[T](t.shape)

proc ones*[T: SomeNumber](shape: varargs[int]): Tensor[T] {.noInit,noSideEffect,inline.} =
  ## Creates a new Tensor filled with 1
  ## Input:
  ##      - Shape of the Tensor
  ##      - Type of its elements
  ## Result:
  ##      - A one-ed Tensor of the same shape
  tensorCpu(shape, result)
  result.data = newSeqWith(result.size, 1.T)

proc ones*[T: SomeNumber](shape: MetadataArray): Tensor[T] {.noInit,noSideEffect,inline.} =
  ## Creates a new Tensor filled with 1
  ## Input:
  ##      - Shape of the Tensor
  ##      - Type of its elements
  ## Result:
  ##      - A one-ed Tensor of the same shape
  tensorCpu(shape, result)
  result.data = newSeqWith(result.size, 1.T)

proc ones_like*[T: SomeNumber](t: Tensor[T]): Tensor[T] {.noInit,noSideEffect, inline.} =
  ## Creates a new Tensor filled with 1 with the same shape as the input
  ## and filled with 1
  ## Input:
  ##      - Tensor
  ## Result:
  ##      - A one-ed Tensor of the same shape
  return ones[T](t.shape)

template randomTensorCpu[T](t: Tensor[T], shape: varargs[int], max_or_range: typed): untyped =
  tensorCpu(shape, t)
  t.data = newSeqWith(t.size, T(random(max_or_range))) # Due to automatic converter (float32 -> float64), we must force T #68

proc randomTensor*[T:SomeReal](shape: varargs[int], max: T): Tensor[T] {.noInit.} =
  ## Creates a new float Tensor filled with values between 0 and max.
  ##
  ## Random seed can be set by importing ``random`` and ``randomize(seed)``
  ## Input:
  ##      - a shape
  ##      - the max value possible (float)
  ##      - a tensor backend
  ## Result:
  ##      - A tensor of the input shape filled with random value between 0 and max input value
  randomTensorCpu(result, shape, max)

proc randomTensor*(shape: varargs[int], max: int): Tensor[int] {.noInit.} =
  ## Creates a new int Tensor filled with values between 0 and max-1.
  ##
  ## Random seed can be set by importing ``random`` and ``randomize(seed)``
  ## Input:
  ##      - a shape
  ##      - the max value possible (integer, exclusive)
  ##      - a tensor backend
  ## Result:
  ##      - A tensor of the input shape filled with random value between 0 and max input value (excluded)
  randomTensorCpu(result, shape, max)

proc randomTensor*[T](shape: varargs[int], slice: Slice[T]): Tensor[T] {.noInit.} =
  ## Creates a new int Tensor filled with values in the Slice range.
  ##
  ## Random seed can be set by importing ``random`` and ``randomize(seed)``
  ## Input:
  ##      - a shape
  ##      - a range/slice
  ##      - a tensor backend
  ## Result:
  ##      - A tensor of the input shape filled with random value in the slice range
  randomTensorCpu(result, shape, slice)

proc randomNormal(mean = 0.0, std = 1.0): float =
  ## Random number in the normal distribution using Box-Muller method
  ## See https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform
  var valid {.global.} = false
  var x {.global.}, y {.global.}, rho {.global.}: float
  if not valid:
    x = random(1.0)
    y = random(1.0)
    rho = sqrt(-2.0 * ln(1.0 - y))
    valid = true
    return rho*cos(2.0*PI*x)*std+mean
  else:
    valid = false
    return rho*sin(2.0*PI*x)*std+mean

proc randomNormalTensor*[T:SomeReal](shape: varargs[int], mean:T = 0, std:T = 1): Tensor[T] {.noInit.} =
  ## Creates a new Tensor filled with values in the normal distribution
  ##
  ## Random seed can be set by importing ``random`` and ``randomize(seed)``
  ## Input:
  ##      - a shape
  ##      - the mean (default 0)
  ##      - the standard deviation (default 1)
  ## Result:
  ##      - A tensor of the input shape filled with random values in the normal distribution
  tensorCpu(shape, result)
  result.data = newSeqWith(result.size, T(randomNormal(mean.float, std.float)))