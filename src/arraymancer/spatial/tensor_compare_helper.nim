import ../tensor
proc `<`*[T](s1, s2: Tensor[T]): bool =
  ## just an internal comparison of two Tensors, which assumes that the order of two
  ## seqs matters.
  #let s1 = s1C.toTensorNormal
  #let s2 = s2C.toTensorNormal
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
