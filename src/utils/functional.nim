# The MIT License - Mamy André Ratsimbazafy, 2017

# Contrary to other parts of the Arraymancer project,
# functional.nim is licensed under the MIT license, not Apache Public License.

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
  let len = min(a.len, b.len)

  for i in 0..<len:
    yield (a[i], b[i])

iterator zipWith[T1,T2,T3](f: proc(u: T1, v:T2): T3, a: openarray[T1], b: openarray[T2]): T3  {.inline.} =
  for i in zip(a,b):
    yield f(a,b)