# Laser
# Copyright (c) 2018 Mamy André-Ratsimbazafy
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

import
    ./gemm_ukernel_generator, ./gemm_tiling,
    ../../simd

# mark as used to avoid unused import warnings
{.used.}

x86only()

## For the C codegen of AVX512 instructions to be valid, we need the following flag:
when defined(avx512) and (defined(gcc) or defined(clang)):
  {.passC: "-mavx512dq".}
## See: https://stackoverflow.com/a/63711952
## for a script to find the required compilation flags for specific SIMD functions.

ukernel_generator(
    x86_AVX512,
    typ = float32,
    vectype = m512,
    nb_scalars = 16,
    simd_setZero = mm512_setzero_ps,
    simd_broadcast_value = mm512_set1_ps,
    simd_load_aligned = mm512_load_ps,
    simd_load_unaligned = mm512_loadu_ps,
    simd_store_unaligned = mm512_storeu_ps,
    simd_mul = mm512_mul_ps,
    simd_add = mm512_add_ps,
    simd_fma = mm512_fmadd_ps
  )

ukernel_generator(
    x86_AVX512,
    typ = float64,
    vectype = m512d,
    nb_scalars = 8,
    simd_setZero = mm512_setzero_pd,
    simd_broadcast_value = mm512_set1_pd,
    simd_load_aligned = mm512_load_pd,
    simd_load_unaligned = mm512_loadu_pd,
    simd_store_unaligned = mm512_storeu_pd,
    simd_mul = mm512_mul_pd,
    simd_add = mm512_add_pd,
    simd_fma = mm512_fmadd_pd
  )

template int32x16_muladd_unfused_avx512(a, b, c: m512i): m512i =
  mm512_add_epi32(mm512_mullo_epi32(a, b), c)

ukernel_generator(
    x86_AVX512,
    typ = int32,
    vectype = m512i,
    nb_scalars = 16,
    simd_setZero = mm512_setzero_si512,
    simd_broadcast_value = mm512_set1_epi32,
    simd_load_aligned = mm512_load_si512,
    simd_load_unaligned = mm512_loadu_si512,
    simd_store_unaligned = mm512_storeu_si512,
    simd_mul = mm512_mullo_epi32,
    simd_add = mm512_add_epi32,
    simd_fma = int32x16_muladd_unfused_avx512
    )

template int64x8_muladd_unfused_avx512(a, b, c: m512i): m512i =
  mm512_add_epi64(mm512_mullo_epi64(a, b), c)

ukernel_generator(
    x86_AVX512,
    typ = int64,
    vectype = m512i,
    nb_scalars = 8,
    simd_setZero = mm512_setzero_si512,
    simd_broadcast_value = mm512_set1_epi64,
    simd_load_aligned = mm512_load_si512,
    simd_load_unaligned = mm512_loadu_si512,
    simd_store_unaligned = mm512_storeu_si512,
    simd_mul = mm512_mullo_epi64,
    simd_add = mm512_add_epi64,
    simd_fma = int64x8_muladd_unfused_avx512
    )
