import macros, sequtils


macro getSubType*(TT: typedesc): untyped =
  # Get the subtype T of an AnyTensor[T] input
  getTypeInst(TT)[1][1]


## The following should not be useful, if ops is possible in fprop
## Shape should be matching in bprop and we shoudln't need special scalar treatment

# proc isScalar(t: AnyTensor): bool {.inline.}=
#   for dim in t.shape:
#     if dim != 1 and dim != 0:
#       return false
#   return true


template product*[T: SomeNumber](s: seq[T]): T =
  ## Get the product of all numbers in a sequence or array
  s.foldl(a*b)