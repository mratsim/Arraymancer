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

proc newTensor*(shape: openarray[int], T: typedesc, backend: static[Backend]): auto {.noSideEffect, deprecated.} =
  ## DEPRECATED: For an easier to maintain code (no polymorphic output zeros(..., Cpu) -> Tensor, zeros(Cuda) -> CudaTensor),
  ## init procs will not offer the backend parameter anymore.
  ## Full rationale in the Design_Document on Github.
  ##
  ## Creates a new Tensor
  ## Input:
  ##      - Shape of the Tensor
  ##      - Type of its elements
  ##      - Backend
  ## Result:
  ##      - A Tensor of the proper shape initialized with
  ##        the default type value (0 for numeric types)

  # TODO: Cpu backend as default, pending: https://github.com/nim-lang/Nim/issues/6339

  when backend == Cpu:
    var t: Tensor[T]
    tensorCpu(shape, t)
    t.data = newSeq[T](t.shape.product)
    return t
  elif backend == Cuda:
    var t: CudaTensor[T]
    tensorCuda[T](shape, t)
    return t

proc toTensor*(s:openarray, backend: static[Backend]): auto {.noSideEffect, deprecated.} =
  ## DEPRECATED: For an easier to maintain code (no polymorphic output zeros(..., Cpu) -> Tensor, zeros(Cuda) -> CudaTensor),
  ## init procs will not offer the backend parameter anymore.
  ## Full rationale in the Design_Document on Github.
  ##
  ## Convert an openarray to a Tensor
  # TODO: have Backend.Cpu as default. pending https://github.com/nim-lang/Nim/issues/6339
  when backend == Cpu:
    toTensorCpu(s)

proc toTensor*(s:string, backend: static[Backend]): auto {.noSideEffect, deprecated.} =
  ## DEPRECATED: For an easier to maintain code (no polymorphic output zeros(..., Cpu) -> Tensor, zeros(Cuda) -> CudaTensor),
  ## init procs will not offer the backend parameter anymore.
  ## Full rationale in the Design_Document on Github.
  ##
  ## Convert an openarray to a Tensor
  ##
  ## Handle string specifically (otherwise they are interpreted as openarray[char])
  when backend == Cpu:
    toTensorCpu(s)

# TODO add tests for zeros, ones and randomTensor
proc zeros*[T: SomeNumber](shape: openarray[int], typ: typedesc[T], backend: static[Backend]): auto {.deprecated, noSideEffect, inline.} =  ## Creates a new Tensor filled with 0
  ## DEPRECATED: For an easier to maintain code (no polymorphic output zeros(..., Cpu) -> Tensor, zeros(Cuda) -> CudaTensor),
  ## init procs will not offer the backend parameter anymore.
  ## Full rationale in the Design_Document on Github.
  ##
  ## Input:
  ##      - Shape of the Tensor
  ##      - Type of its elements
  ##      - Backend
  ## Result:
  ##      - A zero-ed Tensor of the input shape
  return newTensor(shape, typ, backend)

proc ones*[T: SomeNumber](shape: openarray[int], typ: typedesc[T], backend: static[Backend]): auto {.deprecated, noSideEffect.} =
  ## DEPRECATED: For an easier to maintain code (no polymorphic output zeros(..., Cpu) -> Tensor, zeros(Cuda) -> CudaTensor),
  ## init procs will not offer the backend parameter anymore.
  ## Full rationale in the Design_Document on Github.
  ##
  ## Creates a new Tensor filled with 1
  ## Input:
  ##      - Shape of the Tensor
  ##      - Type of its elements
  ##      - Backend
  ## Result:
  ##      - A one-ed Tensor of the same shape
  when backend == Cpu:
    var t: Tensor[T]
    tensor(shape, t)
    t.data = newSeqWith(t.shape.product, 1.T)

template randomTensorCpu[T](t: Tensor[T], shape: openarray[int], max_or_range: typed): untyped =
  tensorCpu(shape, t)
  t.data = newSeqWith(t.shape.product, random(max_or_range))

proc randomTensor*(shape: openarray[int], max: float, backend: static[Backend]): auto {.deprecated.}=
  ## DEPRECATED: For an easier to maintain code (no polymorphic output zeros(..., Cpu) -> Tensor, zeros(Cuda) -> CudaTensor),
  ## init procs will not offer the backend parameter anymore.
  ## Full rationale in the Design_Document on Github.
  ##
  ## Creates a new float Tensor filled with values between 0 and max
  ## Random seed can be set by importing ``random`` and ``randomize(seed)``
  ## Input:
  ##      - a shape
  ##      - the max value possible (float)
  ##      - a tensor backend
  ## Result:
  ##      - A tensor of the input shape filled with random value between 0 and max input value
  when backend == Cpu:
    var t: Tensor[float]
    randomTensorCpu(t, shape, max)
    return t

proc randomTensor*(shape: openarray[int], max: int, backend: static[Backend]): auto {.deprecated.}=
  ## DEPRECATED: For an easier to maintain code (no polymorphic output zeros(..., Cpu) -> Tensor, zeros(Cuda) -> CudaTensor),
  ## init procs will not offer the backend parameter anymore.
  ## Full rationale in the Design_Document on Github.
  ##
  ## Creates a new int Tensor filled with values between 0 and max-1
  ## Random seed can be set by importing ``random`` and ``randomize(seed)``
  ## Input:
  ##      - a shape
  ##      - the max value possible (integer, exclusive)
  ##      - a tensor backend
  ## Result:
  ##      - A tensor of the input shape filled with random value between 0 and max input value (excluded)
  when backend == Cpu:
    var t: Tensor[int]
    randomTensorCpu(t, shape, max)
    return t

proc randomTensor*[T](shape: openarray[int], slice: Slice[T], B: static[Backend]): auto {.deprecated.}=
  ## DEPRECATED: For an easier to maintain code (no polymorphic output zeros(..., Cpu) -> Tensor, zeros(Cuda) -> CudaTensor),
  ## init procs will not offer the backend parameter anymore.
  ## Full rationale in the Design_Document on Github.
  ##
  ## Creates a new int Tensor filled with values in the Slice range.
  ## Random seed can be set by importing ``random`` and ``randomize(seed)``
  ## Input:
  ##      - a shape
  ##      - a range/slice
  ##      - a tensor backend
  ## Result:
  ##      - A tensor of the input shape filled with random value in the slice range
  when backend == Cpu:
    var t: Tensor[T]
    randomTensorCpu(t, shape, slice)
    return t