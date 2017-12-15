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


# Custom data structure to force alignment of the buffer arrays
type
  BlasBufferArray[T]  = object
    dataRef: ref[ptr T]
    data*: ptr UncheckedArray[T]
    len*: int

proc deallocBlasBufferArray[T](dataRef: ref[ptr T]) =
  if not dataRef[].isNil:
    deallocShared(dataRef[])
    dataRef[] = nil

proc newBlasBuffer[T](size: int): BlasBufferArray[T] =
  ## Create a heap array aligned with FORCE_ALIGN
  new(result.dataRef, deallocBlasBufferArray)

  # Allocate memory, we will move the pointer, if it does not fall at a modulo FORCE_ALIGN boundary
  let address = cast[ByteAddress](allocShared0(sizeof(T) * size + FORCE_ALIGN - 1))

  result.dataRef[] = cast[ptr T](address)
  result.len = size

  if (address and (FORCE_ALIGN - 1)) == 0:
    result.data = cast[ptr UncheckedArray[T]](address)
  else:
    let offset = FORCE_ALIGN - (address and (FORCE_ALIGN - 1))
    let data_start = cast[ptr UncheckedArray[T]](address +% offset)
    result.data = data_start

proc check_index(a: BlasBufferArray, idx: int) {.inline, noSideEffect.}=
  if idx < 0 or idx >= a.len:
    raise newException(IndexError,  "Index out of bounds, index was " & $idx &
                                      " while length of the array is " & $a)

proc `[]`[T](a: BlasBufferArray[T], idx: int): T {.inline, noSideEffect.} =
  when compileOption("boundChecks"):
    a.check_index(idx)
  a.data[idx]

proc `[]`[T](a: var BlasBufferArray[T], idx: int): var T {.inline, noSideEffect.} =
  when compileOption("boundChecks"):
    a.check_index(idx)
  a.data[idx]

proc `[]=`[T](a: var BlasBufferArray[T], idx: int, v: T) {.inline, noSideEffect.} =
  when compileOption("boundChecks"):
    a.check_index(idx)
  a.data[idx] = v