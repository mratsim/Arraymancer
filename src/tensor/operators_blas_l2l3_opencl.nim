# Copyright (c) 2017-2018 the Arraymancer contributors
# Distributed under the Apache v2 License
# (license terms are at http://www.apache.org/licenses/LICENSE-2.0).

import  ./data_structure,
        ./backend/[opencl_backend, metadataArray],
        ./private/[p_init_opencl, p_checks]


template l1l2_blas_Impl(T: typedesc[SomeReal], clblast_gemv_proc: untyped): untyped =
  proc openCL_MV_y_eq_aAx_p_by(
    alpha: T, a, x: ClTensor[T],
    beta: T, y: var ClTensor[T]) =
    # Matrix-Vector: y = alpha A matvecmul x + beta y

    # TODO: remove this contiguous layout constraint
    if not a.isContiguous:
      raise newException(ValueError, "NotImplemented: for now both tensors should be contiguous")

    let
      a_is_rowMajor = a.is_C_contiguous
      layout =  if a_is_rowMajor: CLBlastLayoutRowMajor
                else: CLBlastLayoutColMajor
      lda = if a_is_rowMajor: a.strides[0]
            else: a.strides[1]

    check clblast_gemv_proc(layout, CLBlastTransposeNo, a.shape[0], a.shape[1],
                alpha,
                a.toClPointer, a.offset, lda,
                x.toClpointer, x.offset, x.strides[0],
                beta,
                y.toClpointer, y.offset, y.strides[0],
                unsafeAddr clQueue0, nil)

l1l2_blas_Impl(float32, clblastSgemv)
l1l2_blas_Impl(float64, clblastDgemv)

proc `*`*[T: SomeReal](a, b: ClTensor[T]): ClTensor[T] =
  ## Matrix multiplication (Matrix-Matrix and Matrix-Vector) on CUDA

  assert b.rank == 1, "Only Matrix-Vector product is supported at the moment"

  if a.rank == 2 and b.rank == 1:
    when compileOption("boundChecks"):
      check_matvec(a,b)
    result = newClTensor[T]([a.shape[0]])
    openCL_MV_y_eq_aAx_p_by(1.T,a, b, 0.T, result)
  else: raise newException(ValueError, "Matrix-Matrix or Matrix-Vector multiplication valid only if first Tensor is a Matrix and second is a Matrix or Vector")