# Copyright (c) 2018-Present Mamy André-Ratsimbazafy
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

# Helpers on column-major triangular matrices
import
  ../../tensor/tensor,
  # TODO: can't call newTensorUninit with optional colMajor, varargs breaks it
  ../../private/sequninit,
  ../../tensor/private/p_init_cpu

# TODO: - don't use assert
#       - move as a public proc

# Triangular matrix implementation
# ---------------------------------

proc tri_impl[T](a: Tensor[T], upper: static bool, k: static int): Tensor[T] {.inline.}=
  ## Returns a matrix with elements:
  ## if upper:
  ##   below the k-th diagonal zero-ed
  ## if lower:
  ##   above the k-th diagonal zero-ed

  assert a.rank == 2

  result = newTensorUninit[T](a.shape)

  let
    nrows = a.shape[0]
    ncols = a.shape[1]
    aRowStride = a.strides[0]
    aColStride = a.strides[1]
    dst = result.get_data_ptr()
    src = a.get_data_ptr()

  const
    k = k     # for emit interpolation. TODO: not the proper way see:
    tile = 32 # https://github.com/nim-lang/Nim/issues/12036#issuecomment-524890898
    cmp = when upper: "<"
          else: ">"

  # We use loop-tiling to deal with row/col imbalances
  # with tile size of 32

  # TODO: proper C interpolation
  # Extra line after define to avoid codegen bug with --debugger:native
  {.emit: """

    #define min(a,b) (((a)<(b))?(a):(b))

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < `nrows`; i+=`tile`)
      for (int j = 0; j < `ncols`; j+=`tile`)
        for (int ii = i; ii<min(i+`tile`, `nrows`); ++ii)
          for (int jj = j; jj<min(j+`tile`,`ncols`); ++jj)
            // dst is row-major
            // src is col-major
            if (jj `cmp` ii + `k`) {
              `dst`[ii * `ncols` + jj] = 0;
            } else {
              `dst`[ii * `ncols` + jj] = `src`[ii * `aRowStride` + jj * `aColStride`];
            }
  """.}

proc triu*[T](a: Tensor[T], k: static int = 0): Tensor[T] =
  tri_impl(a, upper = true, k)

proc tril*[T](a: Tensor[T], k: static int = 0): Tensor[T] =
  tri_impl(a, upper = false, k)

proc tril_unit_diag*[T](a: Tensor[T]): Tensor[T] =
  ## Lower-triangular matrix with unit diagonal
  ## For use with getrf which returns L\U matrices
  ## with L a unit diagonal (not returned) and U a non-unit diagonal (present)

  assert a.rank == 2

  # We return as colMajor for further Fortran processing
  # We need to use low-level tensorCpu to work around Varargs + optional colMajor argument
  tensorCpu(a.shape, result, colMajor)
  result.storage.Fdata = newSeqUninit[T](result.size)

  let
    nrows = a.shape[0]
    ncols = a.shape[1]
    aRowStride = a.strides[0]
    aColStride = a.strides[1]
    dst = result.get_data_ptr()
    src = a.get_data_ptr()

  const
    tile = 32 # https://github.com/nim-lang/Nim/issues/12036#issuecomment-524890898

  # We use loop-tiling to deal with row/col imbalances
  # with tile size of 32

  # TODO: proper C interpolation
  # Extra line after define to avoid codegen bug with --debugger:native
  {.emit: """

    #define min(a,b) (((a)<(b))?(a):(b))

    #pragma omp parallel for collapse(2)
    for (int j = 0; j < `ncols`; j+=`tile`)
      for (int i = 0; i < `nrows`; i+=`tile`)
        for (int jj = j; jj<min(j+`tile`,`ncols`); ++jj)
          for (int ii = i; ii<min(i+`tile`, `nrows`); ++ii)
            // dst is col-major
            // src is col-major
            if (jj > ii) {
              `dst`[jj * `nrows` + ii] = 0;
            } else if (jj == ii) {
              `dst`[jj * `nrows` + ii] = 1;
            } else {
              `dst`[jj * `nrows` + ii] = `src`[ii * `aRowStride` + jj * `aColStride`];
            }
  """.}

proc tril_unit_diag_mut*[T](a: var Tensor[T]) =
  ## Lower-triangular matrix with unit diagonal
  ## For use with getrf which returns L\U matrices
  ##
  ## The input upper-half is overwritten with 0
  ## The input diagonal is overwritten with 1
  ## Input must be column major

  assert a.rank == 2
  assert a.is_F_contiguous, "Input must be column major"

  let
    nrows = a.shape[0]
    ncols = a.shape[1]
    A = a.get_data_ptr()

  const
    tile = 32 # https://github.com/nim-lang/Nim/issues/12036#issuecomment-524890898

  # We use loop-tiling to deal with row/col imbalances
  # with tile size of 32

  # TODO: proper C interpolation
  # Extra line after define to avoid codegen bug with --debugger:native
  {.emit: """

    #define min(a,b) (((a)<(b))?(a):(b))

    #pragma omp parallel for collapse(2)
    for (int j = 0; j < `ncols`; j+=`tile`)
      for (int i = 0; i < `nrows`; i+=`tile`)
        for (int jj = j; jj<min(j+`tile`,`ncols`); ++jj)
          for (int ii = i; ii<min(i+`tile`, `nrows`); ++ii)
            // A is col-major
            if (jj > ii) {
              `A`[jj * `nrows` + ii] = 0;
            } else if (jj == ii) {
              `A`[jj * `nrows` + ii] = 1;
            }
            // else keep value
  """.}


# Sanity checks
# ---------------------------------

when isMainModule:
  # Upper triangular
  block:
    let a = [[1, 2, 3],
             [4, 5, 6],
             [7, 8, 9]].toTensor().asContiguous(colMajor, force=true)

    let ua = [[1, 2, 3],
               [0, 5, 6],
               [0, 0, 9]].toTensor()

    doAssert triu(a) == ua

  block: # From numpy doc
    let a = [[ 1, 2, 3],
             [ 4, 5, 6],
             [ 7, 8, 9],
             [10,11,12]].toTensor().asContiguous(colMajor, force=true)

    let ua = [[ 1, 2, 3],
               [ 4, 5, 6],
               [ 0, 8, 9],
               [ 0, 0,12]].toTensor()

    doAssert triu(a, -1) == ua

  # Lower triangular
  block:
    let a = [[1, 2, 3],
             [4, 5, 6],
             [7, 8, 9]].toTensor().asContiguous(colMajor, force=true)

    let la = [[1, 0, 0],
              [4, 5, 0],
              [7, 8, 9]].toTensor()

    doAssert tril(a) == la

  block: # From numpy doc
    let a = [[ 1, 2, 3],
             [ 4, 5, 6],
             [ 7, 8, 9],
             [10,11,12]].toTensor().asContiguous(colMajor, force=true)

    let la = [[ 0, 0, 0],
              [ 4, 0, 0],
              [ 7, 8, 0],
              [10,11,12]].toTensor()

    doAssert tril(a, -1) == la

  # Lower triangular with unit diagonal
  block:
    let a = [[1, 2, 3],
             [4, 5, 6],
             [7, 8, 9]].toTensor().asContiguous(colMajor, force=true)

    let la = [[1, 0, 0],
              [4, 1, 0],
              [7, 8, 1]].toTensor()

    doAssert tril_unit_diag(a) == la

  block:
    let a = [[ 1, 2, 3],
             [ 4, 5, 6],
             [ 7, 8, 9],
             [10,11,12]].toTensor().asContiguous(colMajor, force=true)

    let la = [[ 1, 0, 0],
              [ 4, 1, 0],
              [ 7, 8, 1],
              [10,11,12]].toTensor()

    doAssert tril_unit_diag(a) == la

  # Lower triangular with unit diagonal - in-place mutation
  block:
    var a = [[1, 2, 3],
             [4, 5, 6],
             [7, 8, 9]].toTensor().asContiguous(colMajor, force=true)

    let la = [[1, 0, 0],
              [4, 1, 0],
              [7, 8, 1]].toTensor()

    tril_unit_diag_mut(a)
    doAssert a == la

  block:
    var a = [[ 1, 2, 3],
             [ 4, 5, 6],
             [ 7, 8, 9],
             [10,11,12]].toTensor().asContiguous(colMajor, force=true)

    let la = [[ 1, 0, 0],
              [ 4, 1, 0],
              [ 7, 8, 1],
              [10,11,12]].toTensor()

    tril_unit_diag_mut(a)
    doAssert a == la
