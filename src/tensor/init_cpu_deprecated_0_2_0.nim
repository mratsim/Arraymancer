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