# Copyright (c) 2018-Present Mamy André-Ratsimbazafy
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

import
  ../../laser/tensor/[datatypes, allocator, initialization],
  nimblas

proc newMatrixUninitColMajor*[T](M: var Tensor[T], rows, cols: int) {.noInit, inline.} =
  var size: int
  initTensorMetadata(M, size, [rows, cols], colMajor)
  M.storage.allocCpuStorage(size)
