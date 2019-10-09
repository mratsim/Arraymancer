# Laser
# Copyright (c) 2018 Mamy André-Ratsimbazafy
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

# ############################################################
#
#                    Pointer arithmetics
#
# ############################################################

# Warning for pointer arithmetics be careful of not passing a `var ptr`
# to a function as `var` are passed by hidden pointers in Nim and the wrong
# pointer will be modified. Templates are fine.

func `+`*(p: ptr, offset: int): type(p) {.inline.}=
  ## Pointer increment
  {.emit: "`result` = `p` + `offset`;".}

# ############################################################
#
#  Conversion of the AB auxiliary matrix from SIMD to scalar
#
# ############################################################
import ../../compiler_optim_hints

template to_ptr*(AB: typed, MR, NR: static int, T: typedesc): untyped =
  assume_aligned cast[ptr array[MR, array[NR, T]]](AB[0][0].unsafeaddr)

# ############################################################
#
#                    Matrix View
#
# ############################################################

type
  MatrixView*[T] = object
    buffer*: ptr UncheckedArray[T]
    rowStride*, colStride*: int

func toMatrixView*[T](data: ptr T, rowStride, colStride: int): MatrixView[T] {.inline.} =
  result.buffer = cast[ptr UncheckedArray[T]](data)
  result.rowStride = rowStride
  result.colStride = colStride

template `[]`*[T](view: MatrixView[T], row, col: Natural): T =
  ## Access like a 2D matrix
  view.buffer[row * view.rowStride + col * view.colStride]

template `[]=`*[T](view: MatrixView[T], row, col: Natural, value: T) =
  ## Access like a 2D matrix
  view.buffer[row * view.rowStride + col * view.colStride] = value

func stride*[T](view: MatrixView[T], row, col: Natural): MatrixView[T]{.inline.}=
  ## Returns a new view offset by the row and column stride
  result.buffer = cast[ptr UncheckedArray[T]](
    addr view.buffer[row*view.rowStride + col*view.colStride]
  )
  result.rowStride = view.rowStride
  result.colStride = view.colStride
