macro getSubType(T: typedesc): untyped =
  getTypeInst(T)[1][1]