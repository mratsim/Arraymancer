# Copyright (c) 2018-Present Mamy André-Ratsimbazafy
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

import
  macros,
  ../private/sequninit,
  ../tensor/tensor, nimlapack,
  ../tensor/private/p_init_cpu # TODO: can't call newTensorUninit with optional colMajor, varargs breaks it

# Wrappers for Fortran LAPACK
# We don't use the C interface LAPACKE that
# automatically manages the work arrays:
#   - it's yet another dependency
#   - it cannot deal with strided arrays

# Alias / overload generator
# --------------------------------------------------------------------------------------
macro overload(overloaded_name: untyped, lapack_name: typed{nkSym}): untyped =
  let impl = lapack_name.getImpl()
  impl.expectKind {nnkProcDef, nnkFuncDef}

  # We can't just `result[0] = overloaded_name`
  # as libName (lapack library) is not defined in this scope

  var
    params = @[newEmptyNode()] # No return value for all Lapack proc
    body = newCall(lapack_name)

  impl[3].expectKind nnkFormalParams
  for idx in 1 ..< impl[3].len:
    # Skip arg 0, the return type which is always empty
    params.add impl[3][idx]
    body.add impl[3][idx][0]

  result = newProc(
    name = overloaded_name,
    params = params,
    body = body,
    pragmas = nnkPragma.newTree(ident"inline")
  )

  when false:
    # View proc signature.
    # Some procs like syevr have over 20 parameters
    echo result.toStrLit

# SVD and Eigenvalues/eigenvectors decomposition
# --------------------------------------------------------------------------------------

overload(syevr, ssyevr)
overload(syevr, dsyevr)

proc syevr*[T: SomeFloat](a: Tensor[T], eigenvectors: bool,
  low_idx: int, high_idx: int, result: var tuple[eigenval, eigenvec: Tensor[T]]) =
  ## Wrapper for LAPACK syevr routine (Symmetric Recursive Eigenvalue Decomposition)

  assert a.rank == 2, "Input is not a matrix"
  assert a.shape[0] == a.shape[1], "Input should be a symmetric matrix"
  # TODO, support "symmetric matrices" with only the upper or lower part filled.
  # (Obviously, upper in Fortran is lower in C ...)

  let a = a.clone(colMajor) # Lapack overwrites the input. TODO move optimization

  var
    jobz: cstring
    interval: cstring
    n, lda: int32 = a.shape[0].int32
    uplo: cstring = "U"
    vl, vu: T          # unused: min and max eigenvalue returned
    il, iu: int32      # ranking of the lowest and highest eigenvalues returned
    abstol: T = -1     # Use default. otherwise need to call LAPACK routine dlamch('S') or 2*dlamch('S')
    m, ldz = n
    lwork: int32 = -1  # dimension of a workspace array
    work: seq[T]
    wkopt: T
    liwork: int32 = -1 # dimension of a second workspace array
    iwork: seq[int32]
    iwkopt: int32
    info: int32

  if low_idx == 0 and high_idx == a.shape[0] - 1:
    interval = "A"
  else:
    interval = "I"
    il = int32 low_idx + 1 # Fortran starts indexing with 1
    iu = int32 high_idx + 1
    m = iu - il + 1

  # Setting up output
  var
    isuppz: seq[int32] # unused
    isuppz_ptr: ptr int32
    z: ptr T

  result.eigenval = newTensorUninit[T](a.shape[0]) # Even if less eigenval are selected Lapack requires this much workspace

  if eigenvectors:
    jobz = "V"

    # Varargs + optional colMajor argument issue, must resort to low level proc at the moment
    tensorCpu(ldz.int, m.int, result.eigenvec, colMajor)
    result.eigenvec.storage.Fdata = newSeqUninit[T](result.eigenvec.size)

    z = result.eigenvec.get_data_ptr
    if interval == "A": # or (il == 1 and iu == n): -> Already checked before
      isuppz = newSeqUninit[int32](2*m)
      isuppz_ptr = isuppz[0].addr
  else:
    jobz = "N"

  let w = result.eigenval.get_data_ptr

  # Querying workspaces sizes
  syevr(jobz, interval, uplo, n.addr, a.get_data_ptr, lda.addr, vl.addr, vu.addr, il.addr, iu.addr,
        abstol.addr, m.addr, w, z, ldz.addr, isuppz_ptr, wkopt.addr, lwork.addr, iwkopt.addr, liwork.addr, info.addr)

  # Allocating workspace
  lwork = wkopt.int32
  work = newSeqUninit[T](lwork)
  liwork = iwkopt.int32
  iwork = newSeqUninit[int32](liwork)

  syevr(jobz, interval, uplo, n.addr, a.get_data_ptr, lda.addr, vl.addr, vu.addr, il.addr, iu.addr,
        abstol.addr, m.addr, w, z, ldz.addr, isuppz_ptr, work[0].addr, lwork.addr, iwork[0].addr, liwork.addr, info.addr)

  # Keep only the selected eigenvals
  result.eigenval = result.eigenval[0 ..< m.int]

  when compileOption("boundChecks"):
    if eigenvectors:
      assert m.int == result.eigenvec.shape[1]

  if unlikely(info > 0):
    # TODO, this should not be an exception, not converging is something that can happen and should
    # not interrupt the program. Alternative. Fill the result with Inf?
    raise newException(ValueError, "the algorithm for computing eigenvalues failed to converge")

  if unlikely(info < 0):
    raise newException(ValueError, "Illegal parameter in symeig: " & $(-info))
