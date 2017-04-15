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

template matmul_blas[B;T: SomeReal](a, b, result: Tensor[B,T]): auto =
    # TODO: static dispatch based on Tensor rank
    # TODO: support a transpose parameter
    # TODO: term rewriting to fuse transpose-multiply
    # TODO: term rewriting to fuse multiply-add
    # TODO: restrict to CPU backend

    let
        rowA = a.shape[0]
        colA = a.shape[1]
        rowB = b.shape[0]
        colB = b.shape[1]

    when compileOption("boundChecks"):
        if colA != rowB:
            raise newException(IndexError, "Number of columns in the first matrix: " &
                                            $(colA) &
                                            ", must be the same as the number of rows in the second matrix: " &
                                            $(rowB))
    result.data = newSeq[T](rowA * colB)
    result.dimensions = @[colB, rowA]
    result.strides = @[rowA, 1]
    result.offset = addr result.data[0]

    # General Matrix Multiply from nimblas.
    gemm(rowMajor, noTranspose, noTranspose, rowA, colB, rowB, 1, a.offset, colA, b.offset, colB, 0, result.offset, colB)

proc `*`*[B,T](a, b: Tensor[B,T]): Tensor[B,T] {.inline, noSideEffect.} =
    if (a.rank == 2 and b.rank == 2 and T is SomeReal and B == Backend.Cpu):
        matmul_blas(a, b, result)
    else: raise newException(ValueError, "Tensor multiplications, not implemented for ranks other than 2")