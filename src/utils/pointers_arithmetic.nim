# From Jehan: https://forum.nim-lang.org/t/1188#7366

template ptrMath(body: untyped) =
  template `+`[T](p: ptr T, off: int): ptr T =
    cast[ptr type(p[])](cast[ByteAddress](p) +% off * sizeof(p[]))
  
  template `+=`[T](p: ptr T, off: int) =
    p = p + off
  
  template `-`[T](p: ptr T, off: int): ptr T =
    cast[ptr type(p[])](cast[ByteAddress](p) -% off * sizeof(p[]))
  
  template `-=`[T](p: ptr T, off: int) =
    p = p - off
  
  template `[]`[T](p: ptr T, off: int): T =
    (p + off)[]
  
  template `[]=`[T](p: ptr T, off: int, val: T) =
    (p + off)[] = val
  
  body