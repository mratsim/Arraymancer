# Copyright (c) 2019-Present the Arraymancer contributors
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

import
  nimlapack,
  ./overload,
  ../../tensor

# Wrappers for Fortran LAPACK linear equation driver routines `*SV`
# Currently only `*GESV` is wrapped
# TODO: Implement GBSV, GTSV, POSV, PBSV, PTSV, SYSV

overload(gesv, sgesv)
overload(gesv, dgesv)

proc gesv*[T: SomeFloat](a, b: var Tensor[T], pivot_indices: var seq[int32]) =
  ## Wrapper for LAPACK `*gesv` routines
  ## Solve AX = B for general matrix
  ##
  ## In-place version, this will overwrite a and b
  assert a.rank == 2, "a is not a matrix"
  assert a.is_F_contiguous, "a must be column-major"

  assert b.rank == 2, "b is not a matrix"
  assert b.is_F_contiguous, "b must be column-major"

  # Temporaries
  let
    n = a.shape[0].int32 # colMajor in Fortran
    nrhs = b.shape[1].int32
    ldb = b.shape[0].int32

  var info: int32

  gesv(n=n.unsafeAddr, nrhs=nrhs.unsafeAddr, a=a.get_data_ptr,
       lda=n.unsafeAddr, ipiv=pivot_indices[0].addr, b=b.get_data_ptr,
       ldb=ldb.unsafeAddr, info=info.addr)

  if info < 0:
    raise newException(ValueError, "Illegal parameter in linear solve gesv: " & $(-info))
  if info > 0:
    let
      cinfo = $(info - 1) # Fortran arrays are indexed by 1
      msg = "Error: in LU factorization, diagonal U[" & cinfo & "," & cinfo & "] is zero. " &
            "Matrix is singular/non-invertible and solution could not be computed."
    raise newException(ValueError, msg)
