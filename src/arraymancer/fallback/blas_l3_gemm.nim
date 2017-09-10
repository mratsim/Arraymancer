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

# # Reading
# C++ version: https://stackoverflow.com/questions/35620853/how-to-write-a-matrix-matrix-product-that-can-compete-with-eigen
# uBLAS C++: http://www.mathematik.uni-ulm.de/~lehn/test_ublas/session1/page01.html
# Blaze C++: http://www.mathematik.uni-ulm.de/~lehn/test_blaze/session1/page01.html
# Rust BLIS inspired: https://github.com/bluss/matrixmultiply

# ### TODO:
# - OpenMP parallelization
# {.passl: "-fopenmp".} # Issue: Clang OSX does not support openmp
# {.passc: "-fopenmp".} # and the default GCC is actually a link to Clang

# - Loop unrolling  # Currently Nim `unroll` pragma exists but is ignored.
# - Pass `-march=native` to the compiler
# - Align memory # should be automatic
# - Is there a way to get L1/L2 cache size at compile-time
# - Is there a way to get number of registers at compile-time

# Best numbers depend on
# L1, L2, L3 cache and register size

# L1 cache: 32 KB data + 32 KB instructions since Nehalem (per proc)
# L2 cache: 256KB since Nehalem
# X86-64 Register size: 16 registers 128-bit (16 Bytes) wide (SSE2), 256-bit with AVX
# Loading int in AVX registers needs AVX2 support in CPU.
# Everything must be aligned in memory for faster loading in registers.

# Int/float64 takes 4B
# float32 takes 2B
# --> use "when" to parametrize size at compile-time?

const MC = 96
const KC = 256
const NC = 4096

# The following should be bigger (4x8) but somehow it hurts my performance
# It might be because the compiler is not using the large AVX registers by default.
const MR = 2
const NR = 2

#                    Panels of B of size KC * NR resides in L1 cache
const MCKC = MC*KC # A resides in L2 cache
const KCNC = KC*NC # B resides in L3 cache
const MRNR = MR*NR # Work area: Fit in registers


include ./blas_l3_gemm_packing
include ./blas_l3_gemm_aux
include ./blas_l3_gemm_micro_kernel
include ./blas_l3_gemm_macro_kernel

proc newBufferArray[T: SomeNumber](N: static[int], typ: typedesc[T]): ref array[N, T]  {.noSideEffect.} =
  new result
  for i in 0 ..< N:
    result[i] = 0.T

proc gemm_nn_fallback[T](m, n, k: int,
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

  {.pragma: align16, codegenDecl: "$# $# __attribute__((aligned(16)))".}
  var buffer_A{.align16.} = newBufferArray(MCKC, T)
  var buffer_B{.align16.} = newBufferArray(KCNC, T)
  var buffer_C{.align16.} = newBufferArray(MRNR, T)

  if alpha == 0.T or k == 0:
    gescal(m, n, beta, C, offC, incRowC, incColC)
    return

  for j in 0 ..< nb:
    nc =  if (j != nb-1 or mod_nc == 0): NC
          else: mod_nc

    for k in 0 ..< kb:
      kc       =  if (k != kb-1 or mod_kc == 0): KC
                  else: mod_kc
      tmp_beta =  if k == 0: beta
                  else: 1.T

      pack_dim( nc, kc,
                B, k*KC*incRowB + j*NC*incColB + offB,
                incColB, incRowB, NR,
                buffer_B)

      for i in 0 ..< mb:
        mc = if (i != mb-1 or mod_mc == 0): MC
             else: mod_mc

        pack_dim( mc, kc,
                  A, i*MC*incRowA + k*KC*incColA + offA,
                  incRowA, incColA, MR,
                  buffer_A)

        gemm_macro_kernel(mc, nc, kc,
                          alpha, tmp_beta,
                          C, i*MC*incRowC + j*NC*incColC + offC,
                          incRowC, incColC, buffer_A, buffer_B, buffer_C)