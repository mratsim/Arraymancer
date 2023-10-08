# Laser
# Copyright (c) 2018 Mamy André-Ratsimbazafy
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

import ../compiler_optim_hints

func align_raw_data*(T: typedesc, p: pointer): ptr UncheckedArray[T] =
  static: assert T is KnownSupportsCopyMem
  withCompilerOptimHints()

  let address = cast[uint](p)
  let aligned_ptr{.restrict.} = block: # We cannot directly apply restrict to the default "result"
    let remainder = address and (LASER_MEM_ALIGN - 1) # modulo LASER_MEM_ALIGN (power of 2)
    if remainder == 0:
      assume_aligned cast[ptr UncheckedArray[T]](address)
    else:
      let offset = LASER_MEM_ALIGN - remainder
      assume_aligned cast[ptr UncheckedArray[T]](address + offset)
  return aligned_ptr
