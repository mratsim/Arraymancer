# Copyright (c) 2018-Present Mamy André-Ratsimbazafy
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

import
  ../../private/sequninit,
  ../../tensor,
  ../../tensor/private/p_init_cpu # TODO: can't call newTensorUninit with optional colMajor, varargs breaks it

func newMatrixUninitColMajor*[T](M: var Tensor[T], rows, cols: int) {.noInit, inline.} =
  tensorCpu(rows, cols, M, colMajor)
  M.storage.Fdata = newSeqUninit[T](rows*cols)

export tensorCpu
