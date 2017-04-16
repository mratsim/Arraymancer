# Adapted from Jehan: https://forum.nim-lang.org/t/1188#7366
# Added `-` to get integer offset of pointers


template ptrMath(body: untyped) =
  {.push hint[XDeclaredButNotUsed]: off.}
  # XDeclaredButNotUsed pending: https://github.com/nim-lang/Nim/issues/4044

  template `+`[T](p: ptr T, off: int): ptr T =
    cast[ptr type(p[])](cast[ByteAddress](p) +% off * sizeof(p[]))
  
  template `+=`[T](p: ptr T, off: int) =
    p = p + off
  
  # template `-`[T](p: ptr T, off: int): ptr T =
  #   cast[ptr type(p[])](cast[ByteAddress](p) -% off * sizeof(p[]))
  # 
  # template `-=`[T](p: ptr T, off: int) =
  #   p = p - off
  # 
  # template `[]`[T](p: ptr T, off: int): T =
  #   (p + off)[]
  # 
  # template `[]=`[T](p: ptr T, off: int, val: T) =
  #   (p + off)[] = val

  ## TODO: Thorougly test this, especially with negative offset
  template `-`[T](off_p: ptr T, p: ptr T): int =
    (cast[ByteAddress](off_p) -% cast[ByteAddress](p)) div sizeof(p[])

  {.pop.}
  body