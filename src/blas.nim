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


## Linear algebra routines for rank 1 and 2 tensors (vectors and matrices)
# TODO: static dispatch based on Tensor rank
# TODO: support a transpose parameter
# TODO: term rewriting to fuse transpose-multiply
# TODO: term rewriting to fuse multiply-add
# TODO: restrict to CPU backend

#######################################
## Get Row Major / Col Major Convention
## Check if Transpose is needed.
## depends on nimblas rowMajor and colMajor
## In Arraymancer we prefer C convention (Row Major)
## so BLAS result are asked in RowMajor layout

proc getLayout(t: Tensor): OrderType {.inline,noSideEffect.}=
    if is_C_contiguous(t): return OrderType.rowMajor
    elif is_F_contiguous(t): return OrderType.colMajor
    else: raise newException(ValueError,"Operation not supported for this matrix. It has a non-contiguous layout")

proc isTransposeNeeded(t: Tensor): TransposeType {.inline,noSideEffect.}=
    if is_C_contiguous(t): return TransposeType.noTranspose
    elif is_F_contiguous(t): return TransposeType.transpose
    else: raise newException(ValueError,"Operation not supported for this matrix. It has a non-contiguous layout")

############################
## Bounds checking functions
template check_matmat(a, b:Tensor) =
    let colA = a.shape[1]
    let rowB = b.shape[0]

    if colA != rowB:
        raise newException(IndexError, "Number of columns in the first matrix: " &
                                        $(colA) &
                                        ", must be the same as the number of rows in the second matrix: " &
                                        $(rowB))
    if offset_to_index(a) != 0 or offset_to_index(b) != 0:
        raise newException(IndexError, "One of the Matrices has a non-0 offset")

template check_matvec(a, b:Tensor) =
    let colA = a.shape[1]
    let rowB = b.shape[0]

    if colA != rowB:
        raise newException(IndexError, "Number of columns in the matrix: " &
                                        $(colA) &
                                        ", must be the same as the number of rows in the vector: " &
                                        $(rowB))
    if offset_to_index(a) != 0 or offset_to_index(b) != 0:
        raise newException(IndexError, "Matrice and/or Vector have a non-0 offset")

template check_dot_prod(a, b:Tensor) =
    if a.rank != 1 or b.rank != 1: raise newException(ValueError, "Dot product is only supported for vectors (tensors of rank 1)")
    if a.dimensions != b.dimensions: raise newException(ValueError, "Vector should be the same length")
    if offset_to_index(a) != 0 or offset_to_index(b) != 0:
        raise newException(IndexError, "One of the Vectors has a non-0 offset")

template check_add(a, b:Tensor) =
    if a.strides != b.strides:
        raise newException(ValueError, "Both Tensors should have the exact same shape")
    if offset_to_index(a) != 0 or offset_to_index(b) != 0:
        raise newException(IndexError, "One of the Vectors has a non-0 offset")


##########################################################################
## BLAS Level 1 (Vector dot product, Addition, Scalar to Vector/Matrix)
# TODO: Add OpenMP pragma for parallel computing?

proc `.*`*[T: SomeReal](a, b: Tensor[Backend.Cpu,T]): T {.noSideEffect.} =
    ## Vector to Vector dot (scalar) product
    when compileOption("boundChecks"): check_dot_prod(a,b)
    return dot(a.dimensions[0], a.offset, 1, b.offset, 1)

proc `.*`*[T: SomeInteger](a, b: Tensor[Backend.Cpu,T]): T {.noSideEffect.} =
    ## Vector to Vector dot (scalar) product
    # Fallback for non-floats
    when compileOption("boundChecks"): check_dot_prod(a,b)
    for ai, bi in zip(a.data, b.data):
        result += ai * bi

proc `+`*[T: SomeNumber](a, b: Tensor[Backend.Cpu,T]): T {.noSideEffect.} =
    ## Vector to Vector dot (scalar) product
    when compileOption("boundChecks"): check_add(a,b)

    result.data = newSeq[T](a.data.len)
    result.dimensions = a.dimensions
    result.strides = a.strides
    result.offset = addr result.data[0]

    var i = 0
    for ai, bi in zip(a.data, b.data):
        result[i] = ai + bi

proc `-`*[T: SomeNumber](a, b: Tensor[Backend.Cpu,T]): T {.noSideEffect.} =
    ## Vector to Vector dot (scalar) product
    when compileOption("boundChecks"): check_add(a,b)

    result.data = newSeq[T](a.data.len)
    result.dimensions = a.dimensions
    result.strides = a.strides
    result.offset = addr result.data[0]

    var i = 0
    for ai, bi in zip(a.data, b.data):
        result[i] = ai - bi

proc `*`*[T: SomeNumber](a: T, t: Tensor[Backend.Cpu,T]): Tensor[Backend.Cpu,T] {.noSideEffect.} =
    proc f(x: T): T = a * x
    return t.fmap(f)

proc `*`*[T: SomeNumber](t: Tensor[Backend.Cpu,T], a: T): Tensor[Backend.Cpu,T] {.noSideEffect.} =
    proc f(x: T): T = a * x
    return t.fmap(f)

proc `/`*[T: SomeNumber](t: Tensor[Backend.Cpu,T], a: T): Tensor[Backend.Cpu,T] {.noSideEffect.} =
    proc f(x: T): T = x / a
    return t.fmap(f)

####################################################
## BLAS Level 2 and 3 (Matrix-Matrix, Matrix-Vector)

template matmat_blas[T: SomeReal](a, b, result: Tensor[Backend.Cpu,T], a_tr, b_tr: TransposeType): auto =
    ## Matrix to matrix Multiply for float tensors of rank 2 in RowMajor ordering
    let
        rowA = a.shape[0]
        colA = a.shape[1]
        rowB = b.shape[0]
        colB = b.shape[1]

    when compileOption("boundChecks"): check_matmat(a,b)

    result.data = newSeq[T](rowA * colB)
    result.dimensions = @[colB, rowA]
    result.strides = @[rowA, 1]
    result.offset = addr result.data[0]

    # General Matrix Multiply from nimblas.
    gemm(rowMajor, a_tr, b_tr, rowA, colB, rowB, 1, a.offset, colA, b.offset, colB, 0, result.offset, colB)

template matvec_blas[T: SomeReal](a, b, result: Tensor[Backend.Cpu,T], a_tr: TransposeType): auto =
    ## Matrix to Vector Multiply for float tensors of rank 2 and 1 in RowMajor ordering
    let
        rowA = a.shape[0]
        colA = a.shape[1]
        rowB = b.shape[0] # B is considered as a column vector

    when compileOption("boundChecks"): check_matvec(a,b)

    result.data = newSeq[T](rowA)
    result.dimensions = @[rowA]
    result.strides = @[1]
    result.offset = addr result.data[0]

    # General Matrix-Vector Multiply from nimblas.
    gemv(rowMajor, a_tr, rowA, rowB, 1, a.offset, colA, b.offset, 1, 0, result.offset, 1)

template mul_dispatch[T: SomeReal](a, b, res: Tensor[Backend.Cpu,T], a_rank, b_rank: int, a_tr, b_tr: TransposeType): auto =
    ## Dispatch for Matrix/Vector multiplication
    # Input == Check: Rank A, Rank B, A is Row Major, B is Row Major
    if a.rank == 2 and b.rank == 2:    matmat_blas(a, b, res, a_tr, b_tr)
    elif a.rank == 2 and b.rank == 1:  matvec_blas(a, b, result, a_tr)
    else: raise newException(ValueError, "Matrix-Matrix or Matrix-Vector multiplication valid only if first Tensor is a Matrix and second is a Matrix or Vector")

proc `*`*[T: SomeReal](a, b: Tensor[Backend.Cpu,T]): Tensor[Backend.Cpu,T] {.noSideEffect.} =
    ## Matrix multiplication (Matrix-Matrix and Matrix-Vector)
    mul_dispatch(a, b, result, a.rank, b.rank, a.isTransposeNeeded, b.isTransposeNeeded)