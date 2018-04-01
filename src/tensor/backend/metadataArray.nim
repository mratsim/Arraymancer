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

import ./global_config

type DynamicStackArray*[T] = object
    ## Custom stack allocated array that behaves like seq.
    ## We must avoid seq creation when modifying tensor shapes, strides or slicing in a tight loop.
    ## Seq creation are also not possible within an OpenMP loop.
    data*: array[MAXRANK, T]
    len*: int

# On x86-64, a cache line can contain 8 int64. Hence for best performance
# MetadataArray should be an array of 7 elements + 1 int for length

# TODO ensure cache line alignment: pending https://github.com/nim-lang/Nim/issues/5315

type
  MetadataArray* = DynamicStackArray[int]
    ## Custom stack allocated array that holds tensors metadata

proc initMetadataArray*(len: int): MetadataArray {.inline.} =
  result.len = len

proc toMetadataArray*(s: varargs[int]): MetadataArray {.inline.} =
  # boundsChecks automatically done for array indexing
  # when compileOption("boundChecks"):
  #   assert s.len <= MAXRANK
  result.len = s.len
  for i in 0..<s.len:
    result.data[i] = s[i]

proc copyFrom*(a: var DynamicStackArray, s: varargs[int]) {.inline.} =
  # boundsChecks automatically done for array indexing
  # when compileOption("boundChecks"):
  #   assert s.len <= MAXRANK
  a.len = s.len
  for i in 0..<s.len:
    a.data[i] = s[i]

proc copyFrom*(a: var DynamicStackArray, s: DynamicStackArray) {.inline.} =
  # boundsChecks automatically done for array indexing
  # when compileOption("boundChecks"):
  #   assert s.len <= MAXRANK
  a.len = s.len
  for i in 0..<s.len:
    a.data[i] = s.data[i]

proc setLen*(a: var DynamicStackArray, len: int) {.inline.} =
  when compileOption("boundChecks"):
    assert len <= MAXRANK
  a.len = len

proc low*(a: DynamicStackArray): int {.inline.} =
  0

proc high*(a: DynamicStackArray): int {.inline.} =
  a.len-1

type Index = SomeSignedInt or BackwardsIndex
template `^^`(s: DynamicStackArray, i: Index): int =
  when i is BackwardsIndex:
    s.len - int(i)
  else: int(i)

proc `[]`*[T](a: DynamicStackArray[T], idx: Index): T {.inline.} =
  # boundsChecks automatically done for array indexing
  # when compileOption("boundChecks"):
  #   assert idx >= 0 and idx < MAXRANK
  a.data[a ^^ idx]

proc `[]`*[T](a: var DynamicStackArray[T], idx: Index): var T {.inline.} =
  # boundsChecks automatically done for array indexing
  # when compileOption("boundChecks"):
  #   assert idx >= 0 and idx < MAXRANK
  a.data[a ^^ idx]

proc `[]=`*[T](a: var DynamicStackArray[T], idx: Index, v: T) {.inline.} =
  # boundsChecks automatically done for array indexing
  # when compileOption("boundChecks"):
  #   assert idx >= 0 and idx < MAXRANK
  a.data[a ^^ idx] = v

proc `[]`*[T](a: DynamicStackArray[T], slice: Slice[int]): DynamicStackArray[T] {.inline.} =
  let bgn_slice = a ^^ slice.a
  let end_slice = a ^^ slice.b

  if end_slice >= bgn_slice:
    # boundsChecks automatically done for array indexing
    # when compileOption("boundChecks"):
    #   assert slice.a >= 0 and slice.b < a.len
    result.len = (end_slice - bgn_slice + 1)
    for i in 0..<result.len:
      result[i] = a[bgn_slice+i]

iterator items*[T](a: DynamicStackArray[T]): T {.inline.} =
  for i in 0..<a.len:
    yield a.data[i]

iterator mitems*[T](a: var DynamicStackArray[T]): var T {.inline.} =
  for i in 0..<a.len:
    yield a.data[i]

iterator pairs*[T](a: DynamicStackArray[T]): (int, T) {.inline.} =
  for i in 0..<a.len:
    yield (i,a.data[i])

proc `@`*[T](a: DynamicStackArray[T]): seq[T] {.inline.} =
  result = newSeq[int](a.len)
  for i in 0..<a.len:
    result[i] = a.data[i]

proc `$`*(a: DynamicStackArray): string {.inline.} =
  result = "["
  var firstElement = true
  for value in items(a):
    if not firstElement: result.add(", ")
    result.add($value)
    firstElement = false
  result.add("]")

proc product*[T:SomeNumber](a: DynamicStackArray[T]): T {.inline.} =
  result = 1
  for value in items(a):
    result *= value

proc insert*[T](a: var DynamicStackArray[T], value: T, index: int = 0) {.inline.} =
  # boundsChecks automatically done for array indexing
  # when compileOption("boundChecks"):
  #   assert a.len+1 < MAXRANK
  #   assert index >= 0 and index <= a.len
  for i in countdown(a.len, index+1):
    a[i] = a[i-1]
  a[index] = value
  inc a.len

proc delete*(a: var DynamicStackArray, index: int) {.inline.} =
  # boundsChecks automatically done for array indexing
  # when compileOption("boundChecks"):
  #   assert a.len > 0 #TODO: support tensor rank 0
  #   assert index >= 0 and index < a.len

  dec(a.len)
  for i in index..<a.len:
    a[i] = a[i+1]
  a[a.len] = 0

proc add*[T](a: var DynamicStackArray[T], value: T) {.inline.} =
  # boundsChecks automatically done for array indexing
  # when compileOption("boundChecks"):
  #   assert a.len+1 < MAXRANK
  a[a.len] = value
  inc a.len

proc `&`*[T](a: DynamicStackArray[T], value: T): DynamicStackArray[T] {.inline.} =
  result = a
  result.add(value)

proc `&`*(a, b: DynamicStackArray): DynamicStackArray {.inline.} =
  # boundsChecks automatically done for array indexing
  # when compileOption("boundChecks"):
  #   assert a.len+b.len < MAXRANK
  result = a
  result.len += b.len
  for i in 0..<b.len:
    result[a.len + i] = b[i]

proc reversed*(a: DynamicStackArray): DynamicStackArray {.inline.} =
  for i in 0..<a.len:
    result[a.len-i-1] = a[i]
  result.len = a.len

proc reversed*(a: DynamicStackArray, result: var DynamicStackArray) {.inline.} =
  for i in 0..<a.len:
    result[a.len-i-1] = a[i]
  for i in a.len..<result.len:
    result[i] = 0
  result.len = a.len

proc `==`*[T](a: DynamicStackArray[T], s: openarray[T]): bool {.inline.} =
  if a.len != s.len:
    return false
  for i in 0..<s.len:
    if a[i] != s[i]:
      return false
  return true

proc `==`*(a, s: DynamicStackArray): bool {.inline.} =
  if a.len != s.len:
    return false
  for i in 0..<s.len:
    if a[i] != s[i]:
      return false
  return true

iterator zip*[T, U](a: DynamicStackArray[T], b: DynamicStackArray[U]): (T, T)=

  # reshape_no_copy relies on zip stopping early
  let len = min(a.len, b.len)

  for i in 0..<len:
    yield (a[i], b[i])

proc concat*[T](dsas: varargs[DynamicStackArray[T]]): DynamicStackArray[T] =

  var total_len = 0
  for dsa in dsas:
    inc(total_len, dsa.len)

  assert total_len <= MAXRANK

  result.len = total_len
  var i = 0
  for dsa in dsas:
    for val in dsa:
      result[i] = val
      inc(i)

proc max*[T](a: DynamicStackArray[T]): T {.noSideEffect, inline.} =
  for val in a:
    result = max(result, val)
