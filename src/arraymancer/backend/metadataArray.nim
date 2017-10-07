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


type
  MetadataArray* = object
    ## Custom stack allocated array that holds tensors metadata
    data: array[MAXRANK, int]
    len: int

proc newMetadataArray*(len: int): MetadataArray {.inline.} =
  result.len = len

converter toMetadataArray*(s: varargs[int]): MetadataArray {.inline.} =
  # boundsChecks automatically done for array indexing
  # when compileOption("boundChecks"):
  #   assert s.len <= MAXRANK
  result.len = s.len
  for i in 0..<s.len:
    result.data[i] = s[i]

proc copyFrom*(a: var MetadataArray, s: varargs[int]) {.inline.} =
  # boundsChecks automatically done for array indexing
  # when compileOption("boundChecks"):
  #   assert s.len <= MAXRANK
  a.len = s.len
  for i in 0..<s.len:
    a.data[i] = s[i]

proc copyFrom*(a: var MetadataArray, s: MetadataArray) {.inline.} =
  # boundsChecks automatically done for array indexing
  # when compileOption("boundChecks"):
  #   assert s.len <= MAXRANK
  a.len = s.len
  for i in 0..<s.len:
    a.data[i] = s.data[i]

proc setLen*(a: var MetadataArray, len: int) {.inline.} =
  when compileOption("boundChecks"):
    assert len <= MAXRANK
  a.len = len

template len*(a: MetadataArray): int =
  a.len

proc low*(a: MetadataArray): int {.inline.} =
  0

proc high*(a: MetadataArray): int {.inline.} =
  a.len-1

proc `[]`*(a: MetadataArray, idx: int): int {.inline.} =
  # boundsChecks automatically done for array indexing
  # when compileOption("boundChecks"):
  #   assert idx >= 0 and idx < MAXRANK
  a.data[idx]

proc `[]`*(a: var MetadataArray, idx: int): var int {.inline.} =
  # boundsChecks automatically done for array indexing
  # when compileOption("boundChecks"):
  #   assert idx >= 0 and idx < MAXRANK
  a.data[idx]

proc `[]=`*(a: var MetadataArray, idx: int, v: int) {.inline.} =
  # boundsChecks automatically done for array indexing
  # when compileOption("boundChecks"):
  #   assert idx >= 0 and idx < MAXRANK
  a.data[idx] = v

proc `[]`*(a: MetadataArray, slice: Slice[int]): MetadataArray {.inline.} =
  if slice.b >= slice.a:
    # boundsChecks automatically done for array indexing
    # when compileOption("boundChecks"):
    #   assert slice.a >= 0 and slice.b < a.len
    result.len = (slice.b - slice.a + 1)
    for i in 0..result.len:
      result[i] = a[slice.a+i]

iterator items*(a: MetadataArray): int {.inline.} =
  for i in 0..<a.len:
    yield a.data[i]

iterator mitems*(a: var MetadataArray): var int {.inline.} =
  for i in 0..<a.len:
    yield a.data[i]

iterator pairs*(a: MetadataArray): (int, int) {.inline.} =
  for i in 0..<a.len:
    yield (i,a.data[i])

proc `@`*(a: MetadataArray): seq[int] {.inline.} =
  result = newSeq[int](a.len)
  for i in 0..<a.len:
    result[i] = a.data[i]

proc `$`*(a: MetadataArray): string {.inline.} =
  result = "["
  var firstElement = true
  for value in items(a):
    if not firstElement: result.add(", ")
    result.add($value)
    firstElement = false
  result.add("]")

proc product*(a: MetadataArray): int {.inline.} =
  result = 1
  for value in items(a):
    result *= value

proc insert*(a: var MetadataArray, value: int, index: int = 0) {.inline.} =
  # boundsChecks automatically done for array indexing
  # when compileOption("boundChecks"):
  #   assert a.len+1 < MAXRANK
  #   assert index >= 0 and index <= a.len
  for i in countdown(a.len, index+1):
    a[i] = a[i-1]
  a[index] = value
  inc a.len

proc delete*(a: var MetadataArray, index: int) {.inline.} =
  # boundsChecks automatically done for array indexing
  when compileOption("boundChecks"):
    assert a.len > 0
    assert index >= 0 and index < a.len
  for i in countdown(a.len-1, index+1):
    a[i-1] = a[i]
  a[a.len] = 0
  dec a.len

proc add*(a: var MetadataArray, value: int) {.inline.} =
  # boundsChecks automatically done for array indexing
  # when compileOption("boundChecks"):
  #   assert a.len+1 < MAXRANK
  a[a.len] = value
  inc a.len

proc `&`*(a: MetadataArray, value: int): MetadataArray {.inline.} =
  result = a
  result.add(value)

proc `&`*(a: MetadataArray, b: MetadataArray): MetadataArray {.inline.} =
  # boundsChecks automatically done for array indexing
  # when compileOption("boundChecks"):
  #   assert a.len+b.len < MAXRANK
  result = a
  result.len += b.len
  for i in 0..<b.len:
    result[a.len + i] = b[i]

proc reversed*(a: MetadataArray): MetadataArray {.inline.} =
  for i in 0..<a.len:
    result[a.len-i-1] = a[i]
  result.len = a.len

proc reversed*(a: MetadataArray, result: var MetadataArray) {.inline.} =
  for i in 0..<a.len:
    result[a.len-i-1] = a[i]
  for i in a.len..<result.len:
    result[i] = 0
  result.len = a.len

proc `==`*(a: MetadataArray, s: openarray[int]): bool {.inline.} =
  if a.len != s.len:
    return false
  for i in 0..<s.len:
    if a[i] != s[i]:
      return false
  return true

proc `==`*(a: MetadataArray, s: MetadataArray): bool {.inline.} =
  if a.len != s.len:
    return false
  for i in 0..<s.len:
    if a[i] != s[i]:
      return false
  return true

proc `^`*(x: int; a: MetadataArray): int {.inline.} =
  a.len - x

iterator zip(a,b: MetadataArray): (int, int)=
  when compileOption("boundChecks"):
    assert a.len == b.len

  for i in 0..<a.len:
    yield (a[i], b[i])