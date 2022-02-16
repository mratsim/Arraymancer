import ../tensor
proc `<`*[T](s1, s2: Tensor[T]): bool =
  ## just an internal comparison of two Tensors, which assumes that the order of two
  ## seqs matters. This proc is in an extra file, because the proc using it (`queryImpl`)
  ## is generic and we need to call `bind` for this. If it was defined in the same
  ## file, we can't `bind` it for some reason. Further we do not want to export such
  ## a procedure as obviously in the general context this comparison doesn't make sense.
  ## But as we use a HeapQueue of tensors, we *need* a `<` comparison operator.
  doAssert s1.size == s2.size
  result = false
  for i in 0 ..< s1.size:
    if s1[i] == s2[i]:
      # may still be decided, equal up to here
      continue
    elif s1[i] < s2[i]:
      return true
    elif s1[i] > s2[i]:
      return false
