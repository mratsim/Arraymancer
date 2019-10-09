# Laser
# Copyright (c) 2018 Mamy André-Ratsimbazafy
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

import
  ./gemm_ukernel_generator, ./gemm_tiling,
  ../../simd

template float32x8_muladd_unfused(a, b, c: m256): m256 =
  mm256_add_ps(mm256_mul_ps(a, b), c)

template float64x4_muladd_unfused(a, b, c: m256d): m256d =
  mm256_add_pd(mm256_mul_pd(a, b), c)

ukernel_generator(
      x86_AVX,
      typ = float32,
      vectype = m256,
      nb_scalars = 8,
      simd_setZero = mm256_setzero_ps,
      simd_broadcast_value = mm256_set1_ps,
      simd_load_aligned = mm256_load_ps,
      simd_load_unaligned = mm256_loadu_ps,
      simd_store_unaligned = mm256_storeu_ps,
      simd_mul = mm256_mul_ps,
      simd_add = mm256_add_ps,
      simd_fma = float32x8_muladd_unfused
    )

ukernel_generator(
      x86_AVX,
      typ = float64,
      vectype = m256d,
      nb_scalars = 4,
      simd_setZero = mm256_setzero_pd,
      simd_broadcast_value = mm256_set1_pd,
      simd_load_aligned = mm256_load_pd,
      simd_load_unaligned = mm256_loadu_pd,
      simd_store_unaligned = mm256_storeu_pd,
      simd_mul = mm256_mul_pd,
      simd_add = mm256_add_pd,
      simd_fma = float64x4_muladd_unfused
    )
