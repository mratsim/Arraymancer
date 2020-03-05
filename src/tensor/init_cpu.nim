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

import
  # Internal
  ../laser/tensor/[initialization, allocator],
  ../laser/strided_iteration/foreach,
  ./data_structure,
  # Third-party
  nimblas,
  # Standard library
  random,
  math,
  typetraits

export initialization

proc newTensorUninit*[T](shape: varargs[int]): Tensor[T] {.noInit, inline.} =
  ## Creates a new Tensor on Cpu backend
  ## Input:
  ##      - Shape of the Tensor
  ##      - Type of its elements
  ## Result:
  ##      - A Tensor of the proper shape with NO initialization
  ## Warning ⚠
  ##   Tensor data is uninitialized and contains garbage.
  var size: int
  initTensorMetadata(result, size, shape)
  allocCpuStorage(result.storage, size)

proc newTensorUninit*[T](shape: Metadata): Tensor[T] {.noInit, inline.} =
  ## Creates a new Tensor on Cpu backend
  ## Input:
  ##      - Shape of the Tensor
  ##      - Type of its elements
  ## Result:
  ##      - A Tensor of the proper shape with NO initialization
  ## Warning ⚠
  ##   Tensor data is uninitialized and contains garbage.
  var size: int
  initTensorMetadata(result, size, shape)
  allocCpuStorage(result.storage, size)

# newTensor is defined in laser/tensor/initialization

proc newTensorWith*[T](shape: varargs[int], value: T): Tensor[T] {.noInit.} =
  ## Creates a new Tensor filled with the given value
  ## Input:
  ##      - Shape of the Tensor
  ##      - Type of its elements
  ##      - Value to initialize its elements
  ## Result:
  ##      - A Tensor of the proper shape initialized with
  ##        the given value
  # Todo: use a template that can accept proc or value. See the code for newSeqWith: https://github.com/nim-lang/Nim/blob/master/lib/pure/collections/sequtils.nim#L650-L665
  var size: int
  initTensorMetadata(result, size, shape)
  allocCpuStorage(result.storage, size)

  when T.supportsCopyMem:
    forEachContiguous x in result:
      x = value
  else:
    forEachContiguousSerial x in result:
      x = value

proc newTensorWith*[T](shape: Metadata, value: T): Tensor[T] {.noInit.} =
  ## Creates a new Tensor filled with the given value
  ## Input:
  ##      - Shape of the Tensor
  ##      - Type of its elements
  ##      - Value to initialize its elements
  ## Result:
  ##      - A Tensor of the proper shape initialized with
  ##        the given value
  # Todo: use a template that can accept proc or value. See the code for newSeqWith: https://github.com/nim-lang/Nim/blob/master/lib/pure/collections/sequtils.nim#L650-L665
  var size: int
  initTensorMetadata(result, size, shape)
  allocCpuStorage(result.storage, size)

  when T.supportsCopyMem:
    forEachContiguous x in result:
      x = value
  else:
    forEachContiguousSerial x in result:
      x = value

# newTensor is defined in laser/tensor/initialization

proc zeros*[T: SomeNumber|Complex[float32]|Complex[float64]](shape: varargs[int]): Tensor[T] {.noInit, inline.} =
  ## Creates a new Tensor filled with 0
  ##
  ## Input:
  ##      - Shape of the Tensor
  ##      - Type of its elements
  ## Result:
  ##      - A zero-ed Tensor of the input shape on backend Cpu
  result = newTensor[T](shape)

proc zeros*[T: SomeNumber|Complex[float32]|Complex[float64]](shape: Metadata): Tensor[T] {.noInit, inline.} =
  ## Creates a new Tensor filled with 0
  ##
  ## Input:
  ##      - Shape of the Tensor
  ##      - Type of its elements
  ## Result:
  ##      - A zero-ed Tensor of the input shape on backend Cpu
  result = newTensor[T](shape)

proc zeros_like*[T: SomeNumber|Complex[float32]|Complex[float64]](t: Tensor[T]): Tensor[T] {.noInit, inline.} =
  ## Creates a new Tensor filled with 0 with the same shape as the input
  ## Input:
  ##      - Shape of the Tensor
  ##      - Type of its elements
  ## Result:
  ##      - A zero-ed Tensor of the same shape
  result = zeros[T](t.shape)

proc ones*[T: SomeNumber|Complex[float32]|Complex[float64]](shape: varargs[int]): Tensor[T] {.noInit, inline.} =
  ## Creates a new Tensor filled with 1
  ## Input:
  ##      - Shape of the Tensor
  ##      - Type of its elements
  ## Result:
  ##      - A one-ed Tensor of the same shape
  when T is SomeNumber:
    result = newTensorWith[T](shape, 1.T)
  else:
    type F = T.T # Get the float subtype of Complex[T]
    result = newTensorWith[T](shape, complex(1.F, 0.F))

proc ones*[T: SomeNumber|Complex[float32]|Complex[float64]](shape: Metadata): Tensor[T] {.noInit, inline.} =
  ## Creates a new Tensor filled with 1
  ## Input:
  ##      - Shape of the Tensor
  ##      - Type of its elements
  ## Result:
  ##      - A one-ed Tensor of the same shape
  when T is SomeNumber:
    result = newTensorWith[T](shape, 1.T)
  else:
    type F = T.T
    result = newTensorWith[T](shape, complex(1.F, 0.F))

proc ones_like*[T: SomeNumber|Complex[float32]|Complex[float64]](t: Tensor[T]): Tensor[T] {.noInit, inline.} =
  ## Creates a new Tensor filled with 1 with the same shape as the input
  ## and filled with 1
  ## Input:
  ##      - Tensor
  ## Result:
  ##      - A one-ed Tensor of the same shape
  result = ones[T](t.shape)

proc arange*[T: SomeNumber](start, stop, step: T): Tensor[T] {.noInit.} =
  ## Creates a new 1d-tensor with values evenly spaced by ``step``
  ## in the half-open interval [start, stop)
  ##
  ## Resulting size is ceil((stop - start) / step)
  ##
  ## ⚠️ Warnings:
  ## To limit floating point rounding issues, size is computed
  ## by converting to float64.
  ##
  ## - It is recommended to add a small epsilon for non-integer steps
  ## - float64 cannot represent exactly integers over 2^32 (~4.3 billions)
  # TODO: proper exceptions
  assert step != 0, "Step must be non-zero"
  when T is SomeFloat:
    assert start.classify() notin {fcNaN, fcInf, fcNegInf}
    assert stop.classify() notin {fcNaN, fcInf, fcNegInf}
  assert (step > 0 and stop >= start) or (step < 0 and stop <= start), "bounds inconsistent with step sign"

  var size_f64 = ceil((stop.float64 - start.float64) / step.float64)
  assert 0 <= size_f64 and size_f64 <= float64(high(int)), "Invalid size"

  let shape = [int(size_f64)]
  var size: int
  initTensorMetadata(result, size, shape)
  allocCpuStorage(result.storage, size)

  for i in 0 ..< size:
    result.storage.raw_buffer[i] = start + i.T * step

template arange*[T: SomeNumber](stop: T): Tensor[T] =
  # Error messages of templates are very poor
  arange(T(0), stop, T(1))

template arange*[T: SomeNumber](start, stop: T): Tensor[T] =
  # Error messages of templates are very poor
  arange(start, stop, T(1))

template randomTensorCpu[T](t: Tensor[T], shape: varargs[int], max_or_range: typed): untyped =
  var size: int
  initTensorMetadata(result, size, shape)
  allocCpuStorage(result.storage, size)

  for i in 0 ..< t.size:
    result.storage.raw_buffer[i] = T(rand(max_or_range)) # Due to automatic converter (float32 -> float64), we must force T #68

proc randomTensor*[T:SomeFloat](shape: varargs[int], max: T): Tensor[T] {.noInit.} =
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
  ## Creates a new int Tensor filled with values between 0 and max.
  ##
  ## Random seed can be set by importing ``random`` and ``randomize(seed)``
  ## Input:
  ##      - a shape
  ##      - the max value possible (integer, inclusive)
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

proc randomTensor*[T](shape: varargs[int], sample_source: openarray[T]): Tensor[T] {.noInit.} =
  ## Creates a new Tensor filled with values uniformly sampled from ``sample_source``
  ##
  ## Random seed can be set by importing ``random`` and ``randomize(seed)``
  ## Input:
  ##      - a shape
  ##      - a sample_source
  ## Result:
  ##      - A tensor of the input shape filled with random values from ``sample_source``
  var size: int
  initTensorMetadata(result, size, shape)
  allocCpuStorage(result.storage, size)

  for i in 0 ..< size:
    result.storage.raw_buffer[i] = sample(sample_source)

proc randomNormal(mean = 0.0, std = 1.0): float =
  ## Random number in the normal distribution using Box-Muller method
  ## See https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform
  var valid {.global.} = false
  var x {.global.}, y {.global.}, rho {.global.}: float
  if not valid:
    x = rand(1.0)
    y = rand(1.0)
    rho = sqrt(-2.0 * ln(1.0 - y))
    valid = true
    return rho*cos(2.0*PI*x)*std+mean
  else:
    valid = false
    return rho*sin(2.0*PI*x)*std+mean

proc randomNormalTensor*[T:SomeFloat](shape: varargs[int], mean:T = 0, std:T = 1): Tensor[T] {.noInit.} =
  ## Creates a new Tensor filled with values in the normal distribution
  ##
  ## Random seed can be set by importing ``random`` and ``randomize(seed)``
  ## Input:
  ##      - a shape
  ##      - the mean (default 0)
  ##      - the standard deviation (default 1)
  ## Result:
  ##      - A tensor of the input shape filled with random values in the normal distribution
  var size: int
  initTensorMetadata(result, size, shape)
  allocCpuStorage(result.storage, size)

  for i in 0 ..< size:
    result.storage.raw_buffer[i] = T(randomNormal(mean.float, std.float))
