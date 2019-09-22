# Copyright (c) 2018-Present Mamy André-Ratsimbazafy
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

# Helpers on column-major triangular matrices
import
  ../../tensor/tensor

# TODO: - don't use assert
#       - move as a public proc

proc triu*[T](a: Tensor[T], k: static int = 0): Tensor[T] =
  ## Returns a matrix with elements below the k-th diagonal zero-ed

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

  # We use loop-tiling to deal with row/col imbalances
  # with tile size of 32

  {.emit: """
    #define min(a,b) (((a)<(b))?(a):(b))

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < `nrows`; i+=`tile`)
      for (int j = 0; j < `ncols`; j+=`tile`)
        for (int ii = i; ii<min(i+`tile`, `nrows`); ++ii)
          for (int jj = j; jj<min(j+`tile`,`ncols`); ++jj)
            // dst is row-major
            // src is col-major
            if (jj < ii + `k`) {
              dst[ii * `ncols` + jj] = 0;
            } else {
              dst[ii * `ncols` + jj] = src[ii * `aRowStride` + jj * `aColStride`];
            }
  """.}

when isMainModule:

  block:
    let a = [[1, 2, 3],
             [4, 5, 6],
             [7, 8, 9]].toTensor().asContiguous(colMajor, force=true)

    let t_a = [[1, 2, 3],
               [0, 5, 6],
               [0, 0, 9]].toTensor()

    doAssert triu(a) == t_a

  block: # From numpy doc
    let a = [[ 1, 2, 3],
             [ 4, 5, 6],
             [ 7, 8, 9],
             [10,11,12]].toTensor().asContiguous(colMajor, force=true)

    let t_a = [[ 1, 2, 3],
               [ 4, 5, 6],
               [ 0, 8, 9],
               [ 0, 0,12]].toTensor()

    doAssert triu(a, -1) == t_a
