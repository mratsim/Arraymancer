# Copyright 2017 Mamy André-Ratsimbazafy
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

############################
## Bounds checking functions
proc check_matmat(a, b:Tensor) {.noSideEffect.}=
    let colA = a.shape[1]
    let rowB = b.shape[0]

    if colA != rowB:
        raise newException(IndexError, "Number of columns in the first matrix: " &
                                        $(colA) &
                                        ", must be the same as the number of rows in the second matrix: " &
                                        $(rowB))

proc check_matvec(a, b:Tensor)  {.noSideEffect.}=
    let colA = a.shape[1]
    let rowB = b.shape[0]

    if colA != rowB:
        raise newException(IndexError, "Number of columns in the matrix: " &
                                        $(colA) &
                                        ", must be the same as the number of rows in the vector: " &
                                        $(rowB))

proc check_dot_prod(a, b:Tensor)  {.noSideEffect.}=
    if a.rank != 1 or b.rank != 1: raise newException(ValueError, "Dot product is only supported for vectors (tensors of rank 1)")
    if a.shape != b.shape: raise newException(ValueError, "Vector should be the same length")

proc check_add(a, b:Tensor)  {.noSideEffect.}=
    if a.shape != b.shape:
        raise newException(ValueError, "Both Tensors should have the same shape")


##########################################################################
## BLAS Level 1 (Vector dot product, Addition, Scalar to Vector/Matrix)

proc `.*`*[T: SomeReal](a, b: Tensor[Cpu,T]): T {.noSideEffect.} =
    ## Vector to Vector dot (scalar) product
    when compileOption("boundChecks"): check_dot_prod(a,b)
    return dot(a.shape[0], a.get_data_ptr, a.strides[0], b.get_data_ptr, b.strides[0])

proc `.*`*[T: SomeInteger](a, b: Tensor[Cpu,T]): T {.noSideEffect.} =
    ## Vector to Vector dot (scalar) product
    # Fallback for non-floats
    when compileOption("boundChecks"): check_dot_prod(a,b)
    for ai, bi in zip(a.values, b.values):
        result += ai * bi

proc `+`*[T: SomeNumber](a, b: Tensor[Cpu,T]): Tensor[Cpu,T] = # {.noSideEffect.} =
    ## Tensor addition
    when compileOption("boundChecks"): check_add(a,b)

    result.shape = a.shape
    result.strides = shape_to_strides(a.shape)
    result.data = newSeq[T](a.shape.product)
    result.offset = 0

    var i = 0 ## TODO: use pairs/enumerate instead.
    for ai, bi in zip(a.values, b.values):
        result.data[i] = ai + bi
        inc i

proc `-`*[T: SomeNumber](a, b: Tensor[Cpu,T]): Tensor[Cpu,T] {.noSideEffect.} =
    ## Tensor addition
    when compileOption("boundChecks"): check_add(a,b)

    result.shape = a.shape
    result.strides = shape_to_strides(result.shape)
    result.data = newSeq[T](result.shape.product)
    result.offset = 0

    var i = 0 ## TODO: use pairs/enumerate instead.
    for ai, bi in zip(a.values, b.values):
        result.data[i] = ai - bi
        inc i

proc `*`*[T: SomeNumber](a: T, t: Tensor[Cpu,T]): Tensor[Cpu,T] {.noSideEffect.} =
    proc f(x: T): T = a * x
    return t.fmap(f)

proc `*`*[T: SomeNumber](t: Tensor[Cpu,T], a: T): Tensor[Cpu,T] {.noSideEffect.} =
    proc f(x: T): T = a * x
    return t.fmap(f)

proc `/`*[T: SomeNumber](t: Tensor[Cpu,T], a: T): Tensor[Cpu,T] {.noSideEffect.} =
    proc f(x: T): T = x / a
    return t.fmap(f)

####################################################
## BLAS Level 2 and 3 (Matrix-Matrix, Matrix-Vector)

template matmat_blis[T: SomeReal](a, b, result: Tensor[Cpu,T]): auto =
    ## Matrix to matrix Multiply for float tensors of rank 2
    let
        rowA = a.shape[0]
        colA = a.shape[1]
        rowB = b.shape[0]
        colB = b.shape[1]

    when compileOption("boundChecks"): check_matmat(a,b)

    result.data = newSeq[T](rowA * colB)
    result.shape = @[rowA, colB]
    result.strides = @[rowA, 1]  # We force row-major after computation
    result.offset = 0

    let
        a_ptr = get_data_ptr(a)
        b_ptr = get_data_ptr(b)
        res_ptr = get_data_ptr(result)
        alpha = 1.T
        alpha_ptr = unsafeAddr(alpha)
        beta = 0.T
        beta_ptr = unsafeAddr(beta)

    # General stride-aware Matrix Multiply from BLIS.
    bli_gemm(BLIS_NO_TRANSPOSE, BLIS_NO_TRANSPOSE,
            rowA, colB, rowB,
            alpha_ptr,
            a_ptr, a.strides[0], a.strides[1],
            b_ptr, b.strides[0], b.strides[1],
            beta_ptr,
            res_ptr, result.strides[0], 1)

template matmat_blas[T: SomeReal](a, b, result: Tensor[Cpu,T]): auto =
    ## Matrix to matrix Multiply for float tensors of rank 2
    let
        rowA = a.shape[0]
        colA = a.shape[1]
        rowB = b.shape[0]
        colB = b.shape[1]

    when compileOption("boundChecks"): check_matmat(a,b)

    result.data = newSeq[T](rowA * colB)
    result.shape = @[rowA, colB]
    result.strides = @[rowA, 1]
    result.offset = 0

    ## TODO use a GEMM kernel that supports strided arrays like BLIS
    ## That avoids copies and a conversion step
    let
      cont_a = a.asContiguous
      cont_b = b.asContiguous

      a_ptr = get_data_ptr(cont_a)
      b_ptr = get_data_ptr(cont_b)
      res_ptr = get_data_ptr(result)

      a_tr = getTransposeTarget(cont_a)
      b_tr = getTransposeTarget(cont_b)

    # General Matrix Multiply from nimblas.
    if a_tr == TransposeType.noTranspose and b_tr == TransposeType.noTranspose:
        gemm(rowMajor, a_tr, b_tr, rowA, colB, rowB, 1, a_ptr, colA, b_ptr, colB, 0, res_ptr, colB)
    elif a_tr == TransposeType.transpose and b_tr == TransposeType.noTranspose:
        gemm(rowMajor, a_tr, b_tr, rowA, colB, rowB, 1, a_ptr, rowA, b_ptr, colB, 0, res_ptr, colB)
    elif a_tr == TransposeType.noTranspose and b_tr == TransposeType.transpose:
        gemm(rowMajor, a_tr, b_tr, rowA, colB, rowB, 1, a_ptr, colA, b_ptr, rowB, 0, res_ptr, colB)
    elif a_tr == TransposeType.transpose and b_tr == TransposeType.transpose:
        gemm(rowMajor, a_tr, b_tr, rowA, colB, rowB, 1, a_ptr, rowA, b_ptr, rowB, 0, res_ptr, colB)
    else: raise newException(ValueError, "The transposes types: " & $a_tr & " or " & $b_tr & " is not supported")


template matvec_blis[T: SomeReal](a, x, result: Tensor[Cpu,T]): auto =
    ## Matrix to Vector Multiply for float tensors of rank 2 and 1
    let
        rowA = a.shape[0]
        colA = a.shape[1]
        rowX = x.shape[0] # X is considered as a column vector

    when compileOption("boundChecks"): check_matvec(a,b)

    result.data = newSeq[T](rowA)
    result.shape = @[rowA]
    result.strides = @[1]
    result.offset = 0

    let
        a_ptr = get_data_ptr(a)
        x_ptr = get_data_ptr(x)
        res_ptr = get_data_ptr(result)
        alpha = 1.T
        alpha_ptr = unsafeAddr(alpha)
        beta = 0.T
        beta_ptr = unsafeAddr(beta)

    # General stride-aware Matrix-Vector Multiply from BLIS.
    bli_gemv(
            BLIS_NO_TRANSPOSE,
            BLIS_NO_CONJUGATE,
            rowA, rowX,
            alpha_ptr,
            a_ptr, a.strides[0], a.strides[1],
            x_ptr, x.strides[0],
            beta_ptr,
            res_ptr, 1,
            )

template matvec_blas[T: SomeReal](a, x, result: Tensor[Cpu,T]): auto =
    ## Matrix to Vector Multiply for float tensors of rank 2 and 1
    let
        rowA = a.shape[0]
        colA = a.shape[1]
        rowX = x.shape[0] # X is considered as a column vector

    when compileOption("boundChecks"): check_matvec(a,b)

    result.data = newSeq[T](rowA)
    result.shape = @[rowA]
    result.strides = @[1]
    result.offset = 0

    ## TODO use a GEMV kernel that supports strided arrays like BLIS
    ## That avoids copies and a conversion step
    ## Stride for X is supported via incx argument of GEMV
    let cont_a = a.asContiguous

    let a_ptr = get_data_ptr(a)
    let x_ptr = get_data_ptr(x)
    let res_ptr = get_data_ptr(result)

    let a_tr = getTransposeTarget(cont_a)

    # General Matrix-Vector Multiply from nimblas.
    if a_tr == TransposeType.noTranspose: # A is rowMajor
        gemv(rowMajor, a_tr, rowA, rowX, 1, a_ptr, colA, x_ptr, x.strides[0], 0, res_ptr, 1)
    else: # A is colMajor
        gemv(colMajor, noTranspose, rowA, rowX, 1, a_ptr, rowA, x_ptr, x.strides[0], 0, res_ptr, 1)

proc `*`*[T: SomeReal](a, b: Tensor[Cpu,T]): Tensor[Cpu,T] {.noSideEffect.} =
    ## Matrix multiplication (Matrix-Matrix and Matrix-Vector)
    when defined(blis):
        if a.rank == 2 and b.rank == 2:    matmat_blis(a, b, result)
        elif a.rank == 2 and b.rank == 1:  matvec_blis(a, b, result)
        else: raise newException(ValueError, "Matrix-Matrix or Matrix-Vector multiplication valid only if first Tensor is a Matrix and second is a Matrix or Vector")
    else:
        if a.rank == 2 and b.rank == 2:    matmat_blas(a, b, result)
        elif a.rank == 2 and b.rank == 1:  matvec_blas(a, b, result)
        else: raise newException(ValueError, "Matrix-Matrix or Matrix-Vector multiplication valid only if first Tensor is a Matrix and second is a Matrix or Vector")