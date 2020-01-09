# Laser
# Copyright (c) 2018 Mamy André-Ratsimbazafy
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

const LASER_MAXRANK*{.intdefine.} = 6
  # On x86-64, a cache line can contain 8 int64. Hence for best performance
  # MAXRANK should be at most 7 on x86_64 machines.
  # The extra int is used to store the real rank.
  #
  # Alternatively, you may index using int32 which limits dimensions
  # tensor dimensions to 2 billions elements.

type DynamicStackArray*[T] = object
    ## Custom stack allocated array that behaves like seq.
    ## We must avoid seq creation when modifying tensor shapes, strides or slicing in a tight loop.
    ## Seq creation are also not possible within an OpenMP loop.
    data*: array[LASER_MAXRANK, T]
    len*: int

func copyFrom*(a: var DynamicStackArray, s: varargs[int]) =
  a.len = s.len
  for i in 0..<s.len:
    a.data[i] = s[i]

func copyFrom*(a: var DynamicStackArray, s: DynamicStackArray) {.inline.} =
  a = s

func setLen*(a: var DynamicStackArray, len: int) {.inline.} =
  assert len <= MAXRANK
  a.len = len

func low*(a: DynamicStackArray): int {.inline.} =
  0

func high*(a: DynamicStackArray): int {.inline.} =
  a.len-1

type Index = SomeSignedInt or BackwardsIndex
template `^^`(s: DynamicStackArray, i: Index): int =
  when i is BackwardsIndex:
    s.len - int(i)
  else: int(i)

func `[]`*[T](a: DynamicStackArray[T], idx: Index): T {.inline.} =
  a.data[a ^^ idx]

func `[]`*[T](a: var DynamicStackArray[T], idx: Index): var T {.inline.} =
  a.data[a ^^ idx]

func `[]=`*[T](a: var DynamicStackArray[T], idx: Index, v: T) {.inline.} =
  a.data[a ^^ idx] = v

func `[]`*[T](a: DynamicStackArray[T], slice: Slice[int]): DynamicStackArray[T] =
  let bgn_slice = a ^^ slice.a
  let end_slice = a ^^ slice.b

  if end_slice >= bgn_slice:
    result.len = (end_slice - bgn_slice + 1)
    for i in 0..<result.len:
      result[i] = a[bgn_slice+i]

iterator items*[T](a: DynamicStackArray[T]): T =
  for i in 0 ..< a.len:
    yield a.data[i]

iterator mitems*[T](a: var DynamicStackArray[T]): var T =
  for i in 0 ..< a.len:
    yield a.data[i]

iterator pairs*[T](a: DynamicStackArray[T]): (int, T) =
  for i in 0 ..< a.len:
    yield (i,a.data[i])

iterator mpairs*[T](a: var DynamicStackArray[T]): (int, var T) =
  for i in 0 ..< a.len:
    yield (i, a.data[i])

func `@`*[T](a: DynamicStackArray[T]): seq[T] =
  result = newSeq[int](a.len)
  for i in 0..<a.len:
    result[i] = a.data[i]

func `$`*(a: DynamicStackArray): string =
  result = "["
  var firstElement = true
  for value in items(a):
    if not firstElement: result.add(", ")
    result.add($value)
    firstElement = false
  result.add("]")

func product*[T:SomeNumber](a: DynamicStackArray[T]): T =
  result = 1
  for value in items(a):
    result *= value

func insert*[T](a: var DynamicStackArray[T], value: T, index: int = 0) =
  for i in countdown(a.len, index+1):
    a[i] = a[i-1]
  a[index] = value
  inc a.len

func delete*(a: var DynamicStackArray, index: int) =
  dec(a.len)
  for i in index..<a.len:
    a[i] = a[i+1]
  a[a.len] = 0

func add*[T](a: var DynamicStackArray[T], value: T) {.inline.} =
  a[a.len] = value
  inc a.len

func `&`*[T](a: DynamicStackArray[T], value: T): DynamicStackArray[T] {.inline.} =
  result = a
  result.add(value)

func `&`*(a, b: DynamicStackArray): DynamicStackArray =
  result = a
  result.len += b.len
  for i in 0..<b.len:
    result[a.len + i] = b[i]

func reversed*(a: DynamicStackArray): DynamicStackArray =
  for i in 0..<a.len:
    result[a.len-i-1] = a[i]
  result.len = a.len

func reversed*(a: DynamicStackArray, result: var DynamicStackArray) =
  for i in 0..<a.len:
    result[a.len-i-1] = a[i]
  for i in a.len..<result.len:
    result[i] = 0
  result.len = a.len

func `==`*[T](a: DynamicStackArray[T], s: openarray[T]): bool =
  if a.len != s.len:
    return false
  for i in 0..<s.len:
    if a[i] != s[i]:
      return false
  return true

func `==`*(a, s: DynamicStackArray): bool =
  if a.len != s.len:
    return false
  for i in 0..<s.len:
    if a[i] != s[i]:
      return false
  return true

iterator zip*[T, U](a: DynamicStackArray[T], b: DynamicStackArray[U]): (T, T) =
  let len = min(a.len, b.len)

  for i in 0..<len:
    yield (a[i], b[i])

func concat*[T](dsas: varargs[DynamicStackArray[T]]): DynamicStackArray[T] =
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

func max*[T](a: DynamicStackArray[T]): T =
  for val in a:
    result = max(result, val)
