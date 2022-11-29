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

template int32x8_muladd_unfused_avx2(a, b, c: m256i): m256i =
  mm256_add_epi32(mm256_mullo_epi32(a, b), c)

template int32x8_loada(mem_addr: ptr int32): m256i =
  mm256_load_si256(cast[ptr m256i](mem_addr))

template int32x8_loadu(mem_addr: ptr int32): m256i =
  mm256_loadu_si256(cast[ptr m256i](mem_addr))

template int32x8_storeu(mem_addr: ptr int32, a: m256i) =
  mm256_storeu_si256(cast[ptr m256i](mem_addr), a)

ukernel_generator(
      x86_AVX2,
      typ = int32,
      vectype = m256i,
      nb_scalars = 8,
      simd_setZero = mm256_setzero_si256,
      simd_broadcast_value = mm256_set1_epi32,
      simd_load_aligned = int32x8_loada,
      simd_load_unaligned = int32x8_loadu,
      simd_store_unaligned = int32x8_storeu,
      simd_mul = mm256_mullo_epi32,
      simd_add = mm256_add_epi32,
      simd_fma = int32x8_muladd_unfused_avx2
    )
