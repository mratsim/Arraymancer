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


# We can't use manual memory management with the javascript backend

type
  BlasBufferArray[T]  = object
    dataRef: seq[T]
    data*: ptr UncheckedArray[T]
    len*: int

proc newBlasBuffer[T](size: Natural): BlasBufferArray[T] {.inline.}=
  result.dataRef = newSeq[T](size)
  result.data = cast[ptr UncheckedArray[T]](result.dataRef[0].unsafeAddr)
  result.len = size

template assume_aligned[T](data: ptr T, n: csize): ptr T =
  # assume_aligned is a no-op in javascript
  data

proc `[]`[T](a: BlasBufferArray[T], idx: int): T {.inline, noSideEffect.} =
  a.dataRef[idx]

proc `[]`[T](a: var BlasBufferArray[T], idx: int): var T {.inline, noSideEffect.} =
  a.dataRef[idx]

proc `[]=`[T](a: var BlasBufferArray[T], idx: int, v: T) {.inline, noSideEffect.} =
  a.dataRef[idx] = v