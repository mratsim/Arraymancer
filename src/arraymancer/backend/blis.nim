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

type
  ##  Generic error codes
  BlisError {.size: sizeof(cint).} = enum
    BLIS_ERROR_CODE_MAX = (- 140),
    BLIS_EXPECTED_OBJECT_ALIAS = (- 130),
    ##  Object-related errors
    BLIS_ALIGNMENT_NOT_MULT_OF_PTR_SIZE = (- 125),
    BLIS_ALIGNMENT_NOT_POWER_OF_TWO = (- 124),
    BLIS_INSUFFICIENT_STACK_BUF_SIZE = (- 123),
    BLIS_EXHAUSTED_CONTIG_MEMORY_POOL = (- 122),
    BLIS_REQUESTED_CONTIG_BLOCK_TOO_BIG = (- 121),
    BLIS_INVALID_PACKBUF = (- 120),
    ##  Memory allocator errors
    BLIS_EXPECTED_NONNULL_OBJECT_BUFFER = (- 110),
    ##  Buffer-specific errors
    BLIS_PACK_SCHEMA_NOT_SUPPORTED_FOR_UNPACK = (- 100),
    ##  Packing-specific errors
    BLIS_UNEXPECTED_NULL_CONTROL_TREE = (- 90),
    ##  Control tree-specific errors
    BLIS_INVALID_3x3_SUBPART = (- 82),
    BLIS_INVALID_1x3_SUBPART = (- 81),
    BLIS_INVALID_3x1_SUBPART = (- 80),
    ##  Partitioning-specific errors
    BLIS_EXPECTED_UPPER_OR_LOWER_OBJECT = (- 70),
    ##  Storage-specific errors
    BLIS_EXPECTED_TRIANGULAR_OBJECT = (- 63),
    BLIS_EXPECTED_SYMMETRIC_OBJECT = (- 62),
    BLIS_EXPECTED_HERMITIAN_OBJECT = (- 61),
    BLIS_EXPECTED_GENERAL_OBJECT = (- 60),
    ##  Structure-specific errors
    BLIS_INVALID_DIM_STRIDE_COMBINATION = (- 52),
    BLIS_INVALID_COL_STRIDE = (- 51),
    BLIS_INVALID_ROW_STRIDE = (- 50),
    ##  Stride-specific errors
    BLIS_NEGATIVE_DIMENSION = (- 49),
    BLIS_UNEXPECTED_DIAG_OFFSET = (- 48),
    BLIS_UNEXPECTED_VECTOR_DIM = (- 47),
    BLIS_UNEXPECTED_OBJECT_WIDTH = (- 46),
    BLIS_UNEXPECTED_OBJECT_LENGTH = (- 45),
    BLIS_EXPECTED_SQUARE_OBJECT = (- 44),
    BLIS_UNEQUAL_VECTOR_LENGTHS = (- 43),
    BLIS_EXPECTED_VECTOR_OBJECT = (- 42),
    BLIS_EXPECTED_SCALAR_OBJECT = (- 41),
    BLIS_NONCONFORMAL_DIMENSIONS = (- 40),
    ##  Dimension-specific errors
    BLIS_EXPECTED_REAL_VALUED_OBJECT = (- 38),
    BLIS_EXPECTED_REAL_PROJ_OF = (- 37),
    BLIS_INCONSISTENT_DATATYPES = (- 36),
    BLIS_EXPECTED_INTEGER_DATATYPE = (- 35),
    BLIS_EXPECTED_REAL_DATATYPE = (- 34),
    BLIS_EXPECTED_NONCONSTANT_DATATYPE = (- 33),
    BLIS_EXPECTED_NONINTEGER_DATATYPE = (- 32),
    BLIS_EXPECTED_FLOATING_POINT_DATATYPE = (- 31),
    BLIS_INVALID_DATATYPE = (- 30),
    ##  Datatype-specific errors
    BLIS_EXPECTED_NONUNIT_DIAG = (- 26),
    BLIS_INVALID_MACHVAL = (- 25),
    BLIS_INVALID_DIAG = (- 24),
    BLIS_INVALID_CONJ = (- 23),
    BLIS_INVALID_TRANS = (- 22),
    BLIS_INVALID_UPLO = (- 21),
    BLIS_INVALID_SIDE = (- 20),
    ##  Parameter-specific errors
    BLIS_NOT_YET_IMPLEMENTED = (- 13),
    BLIS_NULL_POINTER = (- 12),
    BLIS_UNDEFINED_ERROR_CODE = (- 11),
    BLIS_INVALID_ERROR_CHECKING_LEVEL = (- 10),
    ##  General errors
    BLIS_ERROR_CODE_MIN = (- 9),
    BLIS_FAILURE = (- 2),
    BLIS_SUCCESS = (- 1)


###############################################
### BLIS types
## TODO: check if those shift are constant whatever platforms/architecture Blis is compiled for
const
  BLIS_CONJTRANS_SHIFT = 3
  BLIS_TRANS_SHIFT = 3
  BLIS_CONJ_SHIFT = 4

const
  BLIS_CONJTRANS_BITS = (0x00000003 shl BLIS_CONJTRANS_SHIFT)
  BLIS_TRANS_BIT = (0x00000001 shl BLIS_TRANS_SHIFT)
  BLIS_CONJ_BIT = (0x00000001 shl BLIS_CONJ_SHIFT)

const
  BLIS_BITVAL_TRANS = BLIS_TRANS_BIT
  BLIS_BITVAL_NO_CONJ = 0x00000000
  BLIS_BITVAL_CONJ = BLIS_CONJ_BIT
  BLIS_BITVAL_CONJ_TRANS = (BLIS_CONJ_BIT or BLIS_TRANS_BIT)

type
  BlisTrans {.size: sizeof(cint).} = enum
    BLIS_NO_TRANSPOSE = 0x00000000,
    BLIS_TRANSPOSE = BLIS_BITVAL_TRANS,
    BLIS_CONJ_NO_TRANSPOSE = BLIS_BITVAL_CONJ,
    BLIS_CONJ_TRANSPOSE = BLIS_BITVAL_CONJ_TRANS

type
  BlisConj {.size: sizeof(cint).} = enum
    BLIS_NO_CONJUGATE = 0x00000000,
    BLIS_CONJUGATE = BLIS_BITVAL_CONJ

#################################################

when defined(windows):
  const blisSuffix = ".dll"
else:
  const blisSuffix = ".so" #MacOS & Linux

const libblis = "libblis" & blisSuffix

proc bli_init(): BlisError {.importc: "bli_init", dynlib: libblis.}
proc bli_finalize(): BlisError {.importc: "bli_finalize", dynlib: libblis.}
proc bli_is_initialized(): bool {.importc: "bli_is_initialized", dynlib: libblis.}

proc bli_gemm(
            transa, transb: BlisTrans,
            M, N, K: int,
            alpha: ptr float64,
            A: ptr float64, rsa, csa: int,
            B: ptr float64, rsb, csb: int,
            beta: ptr float64,
            C: ptr float64, rsc, csc: int,
            cntx: ptr int = nil
            )
  {. dynlib: libblis, importc: "bli_dgemm" .}
# C := beta * C + alpha * transa(A) * transb(B)

proc bli_gemm(
            transa, transb: BlisTrans,
            M, N, K: int,
            alpha: ptr float32,
            A: ptr float32, rsa, csa: int,
            B: ptr float32, rsb, csb: int,
            beta: ptr float32,
            C: ptr float32, rsc, csc: int,
            cntx: ptr int = nil
            )
  {. dynlib: libblis, importc: "bli_sgemm" .}
  # Y := beta * Y + alpha * transa(A) * conjx(X)
  # A: Matrix
  # X, Y: Vector

proc bli_gemv(
            transa: BlisTrans,
            conjx: BlisConj,
            M, N: int,
            alpha: ptr float64,
            A: ptr float64, rsa, csa: int,
            X: ptr float64, incx: int,
            beta: ptr float64,
            y: ptr float64, incy: int,
            cntx: ptr int = nil
            )
  {. dynlib: libblis, importc: "bli_dgemv" .}
  # Y := beta * Y + alpha * transa(A) * conjx(X)
  # A: Matrix
  # X, Y: Vector

proc bli_gemv(
            transa: BlisTrans,
            conjx: BlisConj,
            M, N: int,
            alpha: ptr float32,
            A: ptr float32, rsa, csa: int,
            X: ptr float32, incx: int,
            beta: ptr float32,
            y: ptr float32, incy: int,
            cntx: ptr int = nil
            )
  {. dynlib: libblis, importc: "bli_sgemv" .}
  # Y := beta * Y + alpha * transa(A) * conjx(X)
  # A: Matrix
  # X, Y: Vector