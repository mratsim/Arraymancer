when (NimMajor, NimMinor, NimPatch) < (1, 4, 0):
  # IndexDefect was introduced in 1.4.0
  type IndexDefect* = IndexError

when (NimMajor, NimMinor, NimPatch) < (1, 2, 0):
  # csize_t was introduced in 1.2.0
  type csize_t* {.importc: "size_t", nodecl.} = uint

