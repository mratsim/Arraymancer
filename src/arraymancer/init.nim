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

proc fromSeq*[U](s: seq[U], T: typedesc, B: static[Backend]): Tensor[B,T] {.noSideEffect, deprecated.} =
  ## Create a tensor from a nested sequence
  # If sequence is deeply nested Nim cannot detect the very basic type hence U and T in the proc declaration.
  ## DEPRECATED: use toTensor instead
  let shape = s.shape
  let flat = s.flatten

  when compileOption("boundChecks"):
    if (shape.product != flat.len):
        raise newException(IndexError, "Each nested sequence at the same level must have the same number of elements")

  let strides = shape_to_strides(shape)

  result.shape = shape
  result.strides = strides
  result.data = flat
  result.offset = 0