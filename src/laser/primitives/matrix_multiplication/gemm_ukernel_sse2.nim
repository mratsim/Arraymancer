# Laser
# Copyright (c) 2018 Mamy André-Ratsimbazafy
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

import
  ./gemm_ukernel_generator, ./gemm_tiling,
  ../../simd

template float64x2_muladd_unfused(a, b, c: m128d): m128d =
  mm_add_pd(mm_mul_pd(a, b), c)

ukernel_generator(
      x86_SSE2,
      typ = float64,
      vectype = m128d,
      nb_scalars = 2,
      simd_setZero = mm_setzero_pd,
      simd_broadcast_value = mm_set1_pd,
      simd_load_aligned = mm_load_pd,
      simd_load_unaligned = mm_loadu_pd,
      simd_store_unaligned = mm_storeu_pd,
      simd_mul = mm_mul_pd,
      simd_add = mm_add_pd,
      simd_fma = float64x2_muladd_unfused
    )

#######################################
#
# Int32: hack to unroll scalar code
#
#######################################

# This is faster than using the fallback for mm_mullo_epi32
# in laser/primitives/private/sse2_utils

# Note that we are quite limited in registers with scalar code
# as those are competing with all loop controls, conditions, ...

type Int32x2 = array[2, int32]

func setZero_int32_sse2_fallback(): Int32x2 {.inline.} =
  discard

template set1_int32_sse2_fallback(a: int32): Int32x2 =
  [a, a]

func load_int32_sse2_fallback(mem_addr: ptr int32): Int32x2 {.inline.}=
  let p = cast[ptr UncheckedArray[int32]](mem_addr)
  [p[0], p[1]]

func store_int32_sse2_fallback(mem_addr: ptr int32, a: Int32x2) {.inline.}=
  let p = cast[ptr UncheckedArray[int32]](mem_addr)
  p[0] = a[0]
  p[1] = a[1]

template add_int32_sse2_fallback(a, b: Int32x2): Int32x2 =
  [a[0] + b[0], a[1] + b[1]]

template mul_int32_sse2_fallback(a, b: Int32x2): Int32x2 =
  [a[0] * b[0], a[1] * b[1]]

template fma_int32_sse2_fallback(a, b, c: Int32x2): Int32x2 =
  ## By mistake I had c[0] instead of c[1] and twice the speed
  [c[0] + a[0]*b[0], c[1] + a[1]*b[1]]

ukernel_generator(
      x86_SSE2,
      typ = int32,
      vectype = Int32x2,
      nb_scalars = 2,
      simd_setZero = setZero_int32_sse2_fallback,
      simd_broadcast_value = set1_int32_sse2_fallback,
      simd_load_aligned = load_int32_sse2_fallback,
      simd_load_unaligned = load_int32_sse2_fallback,
      simd_store_unaligned = store_int32_sse2_fallback,
      simd_mul = mul_int32_sse2_fallback,
      simd_add = add_int32_sse2_fallback,
      simd_fma = fma_int32_sse2_fallback
    )


#######################################
#
# Int64: hack to unroll scalar code
#
#######################################

type Int64x2 = array[2, int64]

func setZero_int64_sse2_fallback(): Int64x2 {.inline.} =
  discard

template set1_int64_sse2_fallback(a: int64): Int64x2 =
  [a, a]

func load_int64_sse2_fallback(mem_addr: ptr int64): Int64x2 {.inline.}=
  let p = cast[ptr UncheckedArray[int64]](mem_addr)
  [p[0], p[1]]

func store_int64_sse2_fallback(mem_addr: ptr int64, a: Int64x2) {.inline.}=
  let p = cast[ptr UncheckedArray[int64]](mem_addr)
  p[0] = a[0]
  p[1] = a[1]

template add_int64_sse2_fallback(a, b: Int64x2): Int64x2 =
  [a[0] + b[0], a[1] + b[1]]

template mul_int64_sse2_fallback(a, b: Int64x2): Int64x2 =
  [a[0] * b[0], a[1] * b[1]]

template fma_int64_sse2_fallback(a, b, c: Int64x2): Int64x2 =
  ## By mistake I had c[0] instead of c[1] and twice the speed
  [c[0] + a[0]*b[0], c[1] + a[1]*b[1]]

ukernel_generator(
      x86_SSE2,
      typ = int64,
      vectype = Int64x2,
      nb_scalars = 2,
      simd_setZero = setZero_int64_sse2_fallback,
      simd_broadcast_value = set1_int64_sse2_fallback,
      simd_load_aligned = load_int64_sse2_fallback,
      simd_load_unaligned = load_int64_sse2_fallback,
      simd_store_unaligned = store_int64_sse2_fallback,
      simd_mul = mul_int64_sse2_fallback,
      simd_add = add_int64_sse2_fallback,
      simd_fma = fma_int64_sse2_fallback
    )
