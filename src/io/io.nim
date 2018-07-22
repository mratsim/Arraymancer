# Copyright (c) 2018 Mamy André-Ratsimbazafy and the Arraymancer contributors
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

template canImport(x: untyped): bool =
  compiles:
    import x

import ./io_csv, ./io_npy, ./io_image

export io_csv, io_npy, io_image

when canImport(nimhdf5):
  # only provide the hdf5 interface, if the hdf5 library is
  # installed to avoid making `nimhdf5` a dependency
  import ./io_hdf5
  export io_hdf5
