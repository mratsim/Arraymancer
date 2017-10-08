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

# Functional programming and iterator tooling
import sequtils

template scanr*[T](s: seq[T], operation: untyped): untyped =
  ## Template to scan a sequence from right to left, returning the accumulation and intermediate values.
  ## This is a foldr with intermediate steps returned

  ## @[2, 2, 3, 4].scanr(a * b) = @[48, 24, 12, 4]
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

template scanl*[T](s: seq[T], operation: untyped): untyped =
  ## Template to scan a sequence from left to right, returning the accumulation and intermediate values.
  ## This is a foldl with intermediate steps returned

  ## @[2, 2, 3, 4].scanl(a * b) = @[2, 4, 12, 48]
  let len = s.len

  assert len > 0, "Can't scan empty sequences"
  var result = newSeq[T](len)

  result[0] = s[0]
  for i in 1..s.high:
    let
      a {.inject.} = s[i]
      b {.inject.} = result[i-1]
    result[i] = operation
  result

iterator zip*[T1, T2](a: openarray[T1], b: openarray[T2]): (T1,T2) {.noSideEffect.} =
  ## Transform two lists in a list of tuples.
  ## Length of result will be the length of the smallest list, items from the longest will be discarded.
  let len = min(a.len, b.len)

  for i in 0..<len:
    yield (a[i], b[i])

template product*[T: SomeNumber](s: seq[T]): T =
  ## Get the product of all numbers in a sequence or array
  s.foldl(a*b)

proc concatMap*[T](s: seq[T], f: proc(ss: T):string): string  {.noSideEffect.}=
  ## Map a function to a sequence of T and concatenate the result as string
  return s.foldl(a & f(b), "")