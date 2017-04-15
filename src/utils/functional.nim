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

template scanr[T](s: seq[T], operation: untyped): untyped =
  ## Template to scan a sequence from right to left, returning the accumulation and intermediate values.
  ## This is a foldr with intermediate steps returned

  ## @[2, 2, 3, 4].scanr(a + b) = @[48, 24, 12, 4]
  let len = s.len

  assert len > 0, "Can't scan empty sequences"
  var result = newSeq[T](len)

  result[result.high] = s[s.high]
  for i in countdown(len - 1, 1):
    let
      a {.inject.} = s[i-1]
      b {.inject.} = result[i]
    result[i-1] = operation
  result


# zipWith cannot be used with +, * pending: https://github.com/nim-lang/Nim/issues/5702
iterator zip[T1, T2](a: openarray[T1], b: openarray[T2]): (T1,T2) {.inline.} =
  ## Transform two lists in a list of tuples.
  ## Length of result will be the length of the smallest list, items from the longest will be discarded.
  let len = min(a.len, b.len)

  for i in 0..<len:
    yield (a[i], b[i])

iterator zipWith[T1,T2,T3](f: proc(u: T1, v:T2): T3, a: openarray[T1], b: openarray[T2]): seq[T3]  {.inline.} =
  ## Transform two lists in a new one, applying a function to each couple of items from both lists.
  for i in zip(a,b):
    yield f(a,b)

proc zipWith[T1,T2,T3](f: proc(u: T1, v:T2): T3, a: openarray[T1], b: openarray[T2]): seq[T3]  {.inline,noSideEffect.} =
  ## Transform two lists in a new one, applying a function to each couple of items from both lists.
  # We do not call zip for the proc version as that would loop twice
  let m = min(a.len,b.len)
  newSeq(result,m)
  for i in 0..<m: result[i] = f(a[i],b[i])

## Get the product of all numbers in a sequence or array
template product[T: SomeNumber](s: openarray[T]): T = s.foldl(a*b)

## Map a function to a sequence of T and concatenate the result as string
template concatMap[T](s: seq[T], f: proc(ss: T):string): string =
  s.foldl(a & f(b), "")