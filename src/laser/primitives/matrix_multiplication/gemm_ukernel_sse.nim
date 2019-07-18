# Laser
# Copyright (c) 2018 Mamy André-Ratsimbazafy
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

import
  ./gemm_ukernel_generator, ./gemm_tiling,
  ../../simd

template float32x4_muladd_unfused(a, b, c: m128): m128 =
  mm_add_ps(mm_mul_ps(a, b), c)

ukernel_generator(
      x86_SSE,
      typ = float32,
      vectype = m128,
      nb_scalars = 4,
      simd_setZero = mm_setzero_ps,
      simd_broadcast_value = mm_set1_ps,
      simd_load_aligned = mm_load_ps,
      simd_load_unaligned = mm_loadu_ps,
      simd_store_unaligned = mm_storeu_ps,
      simd_mul = mm_mul_ps,
      simd_add = mm_add_ps,
      simd_fma = float32x4_muladd_unfused
    )
