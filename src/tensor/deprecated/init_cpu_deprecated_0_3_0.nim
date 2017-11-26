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


# Deprecated on 2017-09-07 by https://github.com/mratsim/Arraymancer/commit/ea7508c0724a7df7559b68cf8c8470d9ee0d1588
# First release with deprecated tag: 0.2.0

import  ../private/p_init_cpu,
        ../data_structure,
        ../init_cpu,
        sequtils

proc newTensor*(shape: openarray[int], T: typedesc): Tensor[T] {.noSideEffect, inline, deprecated.} =
  ## Creates a new Tensor on Cpu backend
  ## Input:
  ##      - Shape of the Tensor
  ##      - Type of its elements
  ## Result:
  ##      - A Tensor of the proper shape initialized with
  ##        the default type value (0 for numeric types) on Cpu backend
  tensorCpu(shape, result)
  result.data = newSeq[T](result.size)

proc zeros*[T: SomeNumber](shape: openarray[int], typ: typedesc[T]): Tensor[T] {.noSideEffect, inline, deprecated.} =
  ## Creates a new Tensor filled with 0
  ##
  ## Input:
  ##      - Shape of the Tensor
  ##      - Type of its elements
  ## Result:
  ##      - A zero-ed Tensor of the input shape on backend Cpu
  tensorCpu(shape, result)
  result.data = newSeq[T](result.size)


proc ones*[T: SomeNumber](shape: openarray[int], typ: typedesc[T]): Tensor[T] {.noSideEffect,inline, deprecated.} =
  ## Creates a new Tensor filled with 1
  ## Input:
  ##      - Shape of the Tensor
  ##      - Type of its elements
  ## Result:
  ##      - A one-ed Tensor of the same shape
  tensorCpu(shape, result)
  result.data = newSeqWith(result.size, 1.T)

proc unsafeView*[T](t: Tensor[T]): Tensor[T] {.noSideEffect,noInit,inline, deprecated.}=
  ## DEPRECATED: With the switch to reference semantics, ``unsafe`` is now the default.
  ##
  ## Input:
  ##     - A tensor
  ## Returns:
  ##     - A shallow copy. Both tensors share the same memory location.
  ##
  ## Warning ⚠
  ##   Both tensors shares the same memory. Data modification on one will be reflected on the other.
  ##   However modifying the shape, strides or offset will not affect the other.
  result = t

proc unsafeToTensor*[T: SomeNumber](data: seq[T]): Tensor[T] {.noInit,noSideEffect, deprecated.} =
  ## DEPRECATED
  ##
  ## Convert a seq to a Tensor, sharing the seq data
  ## Input:
  ##      - A seq with the tensor data
  ## Result:
  ##      - A rank 1 tensor with the same size of the input
  ## WARNING: result share storage with input
  tensorCpu([data.len], result)
  shallowCopy(result.data, data)