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

# Best numbers depend on
# L1, L2, L3 cache and register size

# L1 cache: 32 KB data + 32 KB instructions since Nehalem (per proc)
# L2 cache: 256KB since Nehalem
# X86-64 Register size: 16 registers 128-bit (16 Bytes) wide (SSE2), 256-bit with AVX
# Loading int in AVX registers needs AVX2 support in CPU.
# Everything must be aligned in memory for faster loading in registers.

# MC must be a multiple of:
# (a) MR (for zero-padding purposes)
# (b) NR (for zero-padding purposes when MR and NR are "swapped")
# NC must be a multiple of
# (a) NR (for zero-padding purposes)
# (b) MR (for zero-padding purposes when MR and NR are "swapped")

# Specific setup for AVX/FMA
const MC = 96
const KC = 128
const NC = 2048

const MR = 4
const NR = 4

#                    Panels of B of size KC * NR resides in L1 cache
const MCKC = MC*KC # A resides in L2 cache
const KCNC = KC*NC # B resides in L3 cache
const MRNR = MR*NR # Work area: Fit in registers

const FORCE_ALIGN = 64

include ./blas_l3_gemm_data_structure
include ./blas_l3_gemm_packing
include ./blas_l3_gemm_aux
include ./blas_l3_gemm_micro_kernel
include ./blas_l3_gemm_macro_kernel

proc gemm_nn_fallback*[T](m, n, k: int,
                alpha: T,
                A: seq[T], offA: int,
                incRowA, incColA: int,
                B: seq[T], offB: int,
                incRowB, incColB: int,
                beta: T,
                C: var seq[T], offC: int,
                incRowC, incColc: int) =

  let
    mb = (m + MC - 1) div MC
    nb = (n + NC - 1) div NC
    kb = (k + KC - 1) div KC

    mod_mc = m mod MC
    mod_nc = n mod NC
    mod_kc = k mod KC

  var mc, nc, kc: int
  var tmp_beta: T

  var buffer_A = newBlasBuffer[T](MCKC)
  var buffer_B = newBlasBuffer[T](KCNC)
  var buffer_C = newBlasBuffer[T](MRNR)

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



##### See blis config: https://github.com/flame/blis/blob/master/config/haswell/bli_kernel.h



#// -- sgemm micro-kernel --

#if 0
#define BLIS_SGEMM_UKERNEL         bli_sgemm_asm_4x24
#define BLIS_DEFAULT_MC_S          256
#define BLIS_DEFAULT_KC_S          256
#define BLIS_DEFAULT_NC_S          4080
#define BLIS_DEFAULT_MR_S          4
#define BLIS_DEFAULT_NR_S          24
#define BLIS_SGEMM_UKERNEL_PREFERS_CONTIG_ROWS
#endif

#if 1
#define BLIS_SGEMM_UKERNEL         bli_sgemm_asm_6x16
#define BLIS_DEFAULT_MC_S          144
#define BLIS_DEFAULT_KC_S          256
#define BLIS_DEFAULT_NC_S          4080
#define BLIS_DEFAULT_MR_S          6
#define BLIS_DEFAULT_NR_S          16

#define BLIS_SGEMM_UKERNEL_PREFERS_CONTIG_ROWS
#endif

#if 0
#define BLIS_SGEMM_UKERNEL         bli_sgemm_asm_16x6
#define BLIS_DEFAULT_MC_S          144
#define BLIS_DEFAULT_KC_S          256
#define BLIS_DEFAULT_NC_S          4080
#define BLIS_DEFAULT_MR_S          16
#define BLIS_DEFAULT_NR_S          6
#endif

#// -- dgemm micro-kernel --

#if 0
#define BLIS_DGEMM_UKERNEL         bli_dgemm_asm_4x12
#define BLIS_DEFAULT_MC_D          152
#define BLIS_DEFAULT_KC_D          160
#define BLIS_DEFAULT_NC_D          4080
#define BLIS_DEFAULT_MR_D          4
#define BLIS_DEFAULT_NR_D          12
#define BLIS_DGEMM_UKERNEL_PREFERS_CONTIG_ROWS
#endif

#if 1
#define BLIS_DGEMM_UKERNEL         bli_dgemm_asm_6x8
#define BLIS_DEFAULT_MC_D          72
#define BLIS_DEFAULT_KC_D          256
#define BLIS_DEFAULT_NC_D          4080
#define BLIS_DEFAULT_MR_D          6
#define BLIS_DEFAULT_NR_D          8

#define BLIS_DGEMM_UKERNEL_PREFERS_CONTIG_ROWS
#endif

#if 0
#define BLIS_DGEMM_UKERNEL         bli_dgemm_asm_8x6
#define BLIS_DEFAULT_MC_D          72
#define BLIS_DEFAULT_KC_D          256
#define BLIS_DEFAULT_NC_D          4080
#define BLIS_DEFAULT_MR_D          8
#define BLIS_DEFAULT_NR_D          6
#endif
