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

template int32x4_muladd_unfused_sse4_1(a, b, c: m128i): m128i =
  mm_add_epi32(mm_mullo_epi32(a, b), c)

template int32x4_loada(mem_addr: ptr int32): m128i =
  mm_load_si128(cast[ptr m128i](mem_addr))

template int32x4_loadu(mem_addr: ptr int32): m128i =
  mm_loadu_si128(cast[ptr m128i](mem_addr))

template int32x4_storeu(mem_addr: ptr int32, a: m128i) =
  mm_storeu_si128(cast[ptr m128i](mem_addr), a)

ukernel_generator(
      x86_SSE4_1,
      typ = int32,
      vectype = m128i,
      nb_scalars = 4,
      simd_setZero = mm_setzero_si128,
      simd_broadcast_value = mm_set1_epi32,
      simd_load_aligned = int32x4_loada,
      simd_load_unaligned = int32x4_loadu,
      simd_store_unaligned = int32x4_storeu,
      simd_mul = mm_mullo_epi32,
      simd_add = mm_add_epi32,
      simd_fma = int32x4_muladd_unfused_sse4_1
    )
