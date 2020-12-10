# Copyright (c) 2017-2018 the Arraymancer contributors
# Distributed under the Apache v2 License
# (license terms are at http://www.apache.org/licenses/LICENSE-2.0).

import  ./data_structure,
        ./backend/opencl_backend,
        ./private/[p_init_opencl, p_checks]


template l1l2_blas_Impl(T: typedesc[SomeFloat], clblast_gemv_proc, clblast_gemm_proc: untyped): untyped =
  proc openCL_MV_y_eq_aAx_p_by(
    alpha: T, a, x: ClTensor[T],
    beta: T, y: var ClTensor[T]) =
    # Matrix-Vector: y = alpha A matvecmul x + beta y

    # TODO: remove this contiguous layout constraint
    if not a.isContiguous:
      raise newException(ValueError, "NotImplemented: for now both tensors should be contiguous")

    let
      a_is_rowMajor = a.is_C_contiguous
      transpose_A = if a_is_rowMajor: CLBlastTransposeNo
                    else: CLBlastTransposeYes
      lda = if a_is_rowMajor: a.strides[0]
            else: a.strides[1]

    check clblast_gemv_proc(CLBlastLayoutRowMajor, transpose_A, a.shape[0], a.shape[1],
                alpha,
                a.toClPointer, a.offset, lda,
                x.toClpointer, x.offset, x.strides[0],
                beta,
                y.toClpointer, y.offset, y.strides[0],
                unsafeAddr clQueue0, nil)

  proc openCL_MM_C_eq_aAB_p_bC(
    alpha: T, a, b: ClTensor[T],
    beta: T, c: var ClTensor[T]) =
    # Matrix: C = alpha A matmul B + beta C

    # TODO: remove this contiguous layout constraint
    if not a.isContiguous:
      raise newException(ValueError, "NotImplemented: for now both tensors should be contiguous")

    assert a.shape[1] == b.shape[0]

    let
      a_is_rowMajor = a.is_C_contiguous
      b_is_rowMajor = b.is_C_contiguous
      c_is_rowMajor = c.is_C_contiguous

      transpose_A = if a_is_rowMajor: CLBlastTransposeNo
                    else: CLBlastTransposeYes
      lda = if a_is_rowMajor: a.strides[0]
            else: a.strides[1]

      transpose_B = if b_is_rowMajor: CLBlastTransposeNo
                    else: CLBlastTransposeYes
      ldb = if b_is_rowMajor: b.strides[0]
            else: b.strides[1]

      layout =  if c_is_rowMajor: CLBlastLayoutRowMajor
                else: CLBlastLayoutColMajor
      ldc = if c_is_rowMajor: c.strides[0]
            else: c.strides[1]

    check clblast_gemm_proc(
      layout, transpose_A, transpose_B,
      a.shape[0], b.shape[1], a.shape[1],
      alpha,
      a.toClpointer, a.offset, lda,
      b.toClpointer, b.offset, ldb,
      beta,
      c.toClpointer, c.offset, ldc,
      clQueue0.unsafeAddr, nil
    )

l1l2_blas_Impl(float32, clblastSgemv, clblastSgemm)
l1l2_blas_Impl(float64, clblastDgemv, clblastDgemm)

proc `*`*[T: SomeFloat](a, b: ClTensor[T]): ClTensor[T] =
  ## Matrix multiplication (Matrix-Matrix and Matrix-Vector) on CUDA

  if a.rank == 2 and b.rank == 2:
    when compileOption("boundChecks"):
      check_matmat(a,b)
    result = newClTensor[T]([a.shape[0], b.shape[1]])
    openCL_MM_C_eq_aAB_p_bC(1.T, a, b, 0.T, result)
  elif a.rank == 2 and b.rank == 1:
    when compileOption("boundChecks"):
      check_matvec(a,b)
    result = newClTensor[T](a.shape[0])
    openCL_MV_y_eq_aAx_p_by(1.T, a, b, 0.T, result)
  else: raise newException(ValueError, "Matrix-Matrix or Matrix-Vector multiplication valid only if first Tensor is a Matrix and second is a Matrix or Vector")
