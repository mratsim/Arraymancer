# Laser
# Copyright (c) 2018 Mamy André-Ratsimbazafy
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

import
  ./gemm_ukernel_generator, ./gemm_tiling,
  ../../simd

x86only()

ukernel_generator(
      x86_AVX_FMA,
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
      simd_fma = mm256_fmadd_ps
    )

ukernel_generator(
      x86_AVX_FMA,
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
      simd_fma = mm256_fmadd_pd,
    )
