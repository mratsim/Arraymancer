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

# The following code is heavily inspired by ulmBLAS (http://apfel.mathematik.uni-ulm.de/~lehn/ulmBLAS/)
# which is heavily inspired by BLIS (https://github.com/flame/blis)
# A big difference (for now?) is instead of passing (const) pointers I pass the (var) array and a var offset.

## Reading
# C++ version: https://stackoverflow.com/questions/35620853/how-to-write-a-matrix-matrix-product-that-can-compete-with-eigen
# uBLAS C++: http://www.mathematik.uni-ulm.de/~lehn/test_ublas/session1/page01.html
# Blaze C++: http://www.mathematik.uni-ulm.de/~lehn/test_blaze/session1/page01.html
# Rust BLIS inspired: https://github.com/bluss/matrixmultiply

#### TODO:
# - OpenMP parallelization
# {.passl: "-fopenmp".} # Issue: Clang OSX does not support openmp
# {.passc: "-fopenmp".} # and the default GCC is actually a link to Clang

# - Loop unrolling  # Currently Nim `unroll` pragma exists but is ignored.
# - Pass `-march=native` to the compiler
# - Align memory # should be automatic
# - Is there a way to get L1/L2 cache size at compile-time
# - Is there a way to get number of registers at compile-time

const MC = 384
const KC = 384
const NC = 4096

const MR = 4
const NR = 4

const MCKC = MC*KC
const KCNC = KC*NC
const MRNR = MR*NR


include ./blas_l3_gemm_packing
include ./blas_l3_gemm_axpy_scal
include ./blas_l3_gemm_micro_kernel
include ./blas_l3_gemm_macro_kernel

proc newBufferArray[T: SomeNumber](N: static[int], typ: typedesc[T]): ref array[N, T]  {.noSideEffect.} =
  new result
  for i in 0 ..< N:
    result[i] = 0.T

# We use T: int so that it is easy to change to float to benchmark against OpenBLAS/MKL/BLIS
proc gemm_nn[T](m, n, k: int,
                alpha: T,
                A: seq[T], offA: int,
                incRowA, incColA: int,
                B: seq[T], offB: int,
                incRowB, incColB: int,
                beta: T,
                C: var seq[T], offC: int,
                incRowC, incColc: int)  {.noSideEffect.} =

  let
    mb = (m + MC - 1) div MC
    nb = (n + NC - 1) div NC
    kb = (k + KC - 1) div KC

    mod_mc = m mod MC
    mod_nc = n mod NC
    mod_kc = k mod KC

  var mc, nc, kc: int
  var tmp_beta: T

  var buffer_A = newBufferArray(MCKC, T)
  var buffer_B = newBufferArray(KCNC, T)
  var buffer_C = newBufferArray(MRNR, T)

  for j in 0 ..< nb:
    nc =  if (j != nb-1 or mod_nc == 0): NC
          else: mod_nc

    for l in 0 ..< kb:
      kc       =  if (l != kb-1 or mod_kc == 0): KC
                  else: mod_kc
      tmp_beta =  if l == 0: beta
                  else: 1.T

      pack_dim( nc, kc,
                B, l*KC*incRowB + j*NC*incColB + offB,
                incColB, incRowB, NR,
                buffer_B)

      for i in 0 ..< mb:
        mc = if (i != mb-1 or mod_mc == 0): MC
             else: mod_mc

        pack_dim( mc, kc,
                  A, i*MC*incRowA+l*KC*incColA + offA,
                  incRowA, incColA, MR,
                  buffer_A)

        gemm_macro_kernel(mc, nc, kc,
                          alpha, tmp_beta,
                          C, i*MC*incRowC + j*NC*incColC + offC,
                          incRowC, incColC, buffer_A, buffer_B, buffer_C)