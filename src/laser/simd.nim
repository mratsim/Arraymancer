# Laser
# Copyright (c) 2018 Mamy André-Ratsimbazafy
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

when defined(i386) or defined(amd64):
  # SIMD throughput and latency:
  #   - https://software.intel.com/sites/landingpage/IntrinsicsGuide/
  #   - https://www.agner.org/optimize/instruction_tables.pdf

  # Reminder: x86 is little-endian, order is [low part, high part]
  # Documentation at https://software.intel.com/sites/landingpage/IntrinsicsGuide/

  when defined(vcc):
    {.pragma: x86_type, byCopy, header:"<intrin.h>".}
    {.pragma: x86, noDecl, header:"<intrin.h>".}
  else:
    {.pragma: x86_type, byCopy, header:"<x86intrin.h>".}
    {.pragma: x86, noDecl, header:"<x86intrin.h>".}
  type
    m128* {.importc: "__m128", x86_type.} = object
      raw: array[4, float32]
    m128d* {.importc: "__m128d", x86_type.} = object
      raw: array[2, float64]
    m128i* {.importc: "__m128i", x86_type.} = object
      raw: array[16, byte]
    m256* {.importc: "__m256", x86_type.} = object
      raw: array[8, float32]
    m256d* {.importc: "__m256d", x86_type.} = object
      raw: array[4, float64]
    m256i* {.importc: "__m256i", x86_type.} = object
      raw: array[32, byte]
    m512* {.importc: "__m512", x86_type.} = object
      raw: array[16, float32]
    m512d* {.importc: "__m512d", x86_type.} = object
      raw: array[8, float64]
    m512i* {.importc: "__m512i", x86_type.} = object
      raw: array[64, byte]
    mmask16* {.importc: "__mmask16", x86_type.} = distinct uint16
    mmask64* {.importc: "__mmask64", x86_type.} = distinct uint64

  # ############################################################
  #
  #                   SSE - float32 - packed
  #
  # ############################################################

  func mm_setzero_ps*(): m128 {.importc: "_mm_setzero_ps", x86.}
  func mm_set1_ps*(a: float32): m128 {.importc: "_mm_set1_ps", x86.}
  func mm_load_ps*(aligned_mem_addr: ptr float32): m128 {.importc: "_mm_load_ps", x86.}
  func mm_loadu_ps*(data: ptr float32): m128 {.importc: "_mm_loadu_ps", x86.}
  func mm_store_ps*(mem_addr: ptr float32, a: m128) {.importc: "_mm_store_ps", x86.}
  func mm_storeu_ps*(mem_addr: ptr float32, a: m128) {.importc: "_mm_storeu_ps", x86.}
  func mm_add_ps*(a, b: m128): m128 {.importc: "_mm_add_ps", x86.}
  func mm_sub_ps*(a, b: m128): m128 {.importc: "_mm_sub_ps", x86.}
  func mm_mul_ps*(a, b: m128): m128 {.importc: "_mm_mul_ps", x86.}
  func mm_max_ps*(a, b: m128): m128 {.importc: "_mm_max_ps", x86.}
  func mm_min_ps*(a, b: m128): m128 {.importc: "_mm_min_ps", x86.}
  func mm_or_ps*(a, b: m128): m128 {.importc: "_mm_or_ps", x86.}

  # ############################################################
  #
  #                    SSE - float32 - scalar
  #
  # ############################################################

  func mm_load_ss*(aligned_mem_addr: ptr float32): m128 {.importc: "_mm_load_ss", x86.}
  func mm_add_ss*(a, b: m128): m128 {.importc: "_mm_add_ss", x86.}
  func mm_max_ss*(a, b: m128): m128 {.importc: "_mm_max_ss", x86.}
  func mm_min_ss*(a, b: m128): m128 {.importc: "_mm_min_ss", x86.}

  func mm_cvtss_f32*(a: m128): float32 {.importc: "_mm_cvtss_f32", x86.}
    ## Extract the low part of the input
    ## Input:
    ##   { A0, A1, A2, A3 }
    ## Result:
    ##   A0

  func mm_movehl_ps*(a, b: m128): m128 {.importc: "_mm_movehl_ps", x86.}
    ## Input:
    ##   { A0, A1, A2, A3 }, { B0, B1, B2, B3 }
    ## Result:
    ##   { B2, B3, A2, A3 }
  func mm_movelh_ps*(a, b: m128): m128 {.importc: "_mm_movelh_ps", x86.}
    ## Input:
    ##   { A0, A1, A2, A3 }, { B0, B1, B2, B3 }
    ## Result:
    ##   { A0, A1, B0, B1 }

  # ############################################################
  #
  #                    SSE2 - float64 - packed
  #
  # ############################################################

  func mm_setzero_pd*(): m128d {.importc: "_mm_setzero_pd", x86.}
  func mm_set1_pd*(a: float64): m128d {.importc: "_mm_set1_pd", x86.}
  func mm_load_pd*(aligned_mem_addr: ptr float64): m128d {.importc: "_mm_load_pd", x86.}
  func mm_loadu_pd*(mem_addr: ptr float64): m128d {.importc: "_mm_loadu_pd", x86.}
  func mm_store_pd*(mem_addr: ptr float64, a: m128d) {.importc: "_mm_store_pd", x86.}
  func mm_storeu_pd*(mem_addr: ptr float64, a: m128d) {.importc: "_mm_storeu_pd", x86.}
  func mm_add_pd*(a, b: m128d): m128d {.importc: "_mm_add_pd", x86.}
  func mm_sub_pd*(a, b: m128d): m128d {.importc: "_mm_sub_pd", x86.}
  func mm_mul_pd*(a, b: m128d): m128d {.importc: "_mm_mul_pd", x86.}

  # ############################################################
  #
  #                    SSE2 - integer - packed
  #
  # ############################################################

  func mm_setzero_si128*(): m128i {.importc: "_mm_setzero_si128", x86.}
  func mm_set1_epi8*(a: int8 or uint8): m128i {.importc: "_mm_set1_epi8", x86.}
  func mm_set1_epi16*(a: int16 or uint16): m128i {.importc: "_mm_set1_epi16", x86.}
  func mm_set1_epi32*(a: int32 or uint32): m128i {.importc: "_mm_set1_epi32", x86.}
  func mm_set1_epi64x*(a: int64 or uint64): m128i {.importc: "_mm_set1_epi64x", x86.}
  func mm_load_si128*(mem_addr: ptr m128i): m128i {.importc: "_mm_load_si128", x86.}
  func mm_loadu_si128*(mem_addr: ptr m128i): m128i {.importc: "_mm_loadu_si128", x86.}
  func mm_storeu_si128*(mem_addr: ptr m128i, a: m128i) {.importc: "_mm_storeu_si128", x86.}
  func mm_add_epi8*(a, b: m128i): m128i {.importc: "_mm_add_epi8", x86.}
  func mm_add_epi16*(a, b: m128i): m128i {.importc: "_mm_add_epi16", x86.}
  func mm_add_epi32*(a, b: m128i): m128i {.importc: "_mm_add_epi32", x86.}
  func mm_add_epi64*(a, b: m128i): m128i {.importc: "_mm_add_epi64", x86.}

  func mm_or_si128*(a, b: m128i): m128i {.importc: "_mm_or_si128", x86.}
  func mm_and_si128*(a, b: m128i): m128i {.importc: "_mm_and_si128", x86.}
  func mm_slli_epi64*(a: m128i, imm8: cint): m128i {.importc: "_mm_slli_epi64", x86.}
    ## Shift 2xint64 left
  func mm_srli_epi64*(a: m128i, imm8: cint): m128i {.importc: "_mm_srli_epi64", x86.}
    ## Shift 2xint64 right
  func mm_srli_epi32*(a: m128i, count: int32): m128i {.importc: "_mm_srli_epi32", x86.}
  func mm_slli_epi32*(a: m128i, count: int32): m128i {.importc: "_mm_slli_epi32", x86.}

  func mm_mullo_epi16*(a, b: m128i): m128i {.importc: "_mm_mullo_epi16", x86.}
    ## Multiply element-wise 2 vectors of 8 16-bit ints
    ## into intermediate 8 32-bit ints, and keep the low 16-bit parts

  func mm_shuffle_epi32*(a: m128i, imm8: cint): m128i {.importc: "_mm_shuffle_epi32", x86.}
    ## Shuffle 32-bit integers in a according to the control in imm8
    ## Formula is in big endian representation
    ## a = {a3, a2, a1, a0}
    ## dst = {d3, d2, d1, d0}
    ## imm8 = {bits76, bits54, bits32, bits10}
    ## d0 will refer a[bits10]
    ## d1            a[bits32]

  func mm_mul_epu32*(a: m128i, b: m128i): m128i {.importc: "_mm_mul_epu32", x86.}
    ## From a = {a1_hi, a1_lo, a0_hi, a0_lo} with a1 and a0 being 64-bit number
    ## and  b = {b1_hi, b1_lo, b0_hi, b0_lo}
    ##
    ## Result = {a1_lo * b1_lo, a0_lo * b0_lo}.
    ## This is an extended precision multiplication 32x32 -> 64

  func mm_set_epi32*(e3, e2, e1, e0: cint): m128i {.importc: "_mm_set_epi32", x86.}
    ## Initialize m128i with {e3, e2, e1, e0} (big endian order)
    ## Storing it will yield [e0, e1, e2, e3]

  func mm_castps_si128*(a: m128): m128i {.importc: "_mm_castps_si128", x86.}
    ## Cast a float32x4 vectors into a 128-bit int vector with the same bit pattern
  func mm_castsi128_ps*(a: m128i): m128 {.importc: "_mm_castsi128_ps", x86.}
    ## Cast a 128-bit int vector into a float32x8 vector with the same bit pattern
  func mm_cvtps_epi32*(a: m128): m128i {.importc: "_mm_cvtps_epi32", x86.}
    ## Convert a float32x4 to int32x4
  func mm_cvtepi32_ps*(a: m128i): m128 {.importc: "_mm_cvtepi32_ps", x86.}
    ## Convert a int32x4 to float32x4

  func mm_cmpgt_epi32*(a, b: m128i): m128i {.importc: "_mm_cmpgt_epi32", x86.}
    ## Compare a greater than b

  func mm_cvtsi128_si32*(a: m128i): cint {.importc: "_mm_cvtsi128_si32", x86.}
    ## Copy the low part of a to int32

  func mm_extract_epi16*(a: m128i, imm8: cint): cint {.importc: "_mm_extract_epi16", x86.}
    ## Extract an int16 from a, selected with imm8
    ## and store it in the lower part of destination (padded with zeroes)

  func mm_movemask_epi8*(a: m128i): int32 {.importc: "_mm_movemask_epi8", x86.}
    ## Returns the most significant bit
    ## of each 8-bit elements in `a`

  # ############################################################
  #
  #                    SSE3 - float32
  #
  # ############################################################

  func mm_movehdup_ps*(a: m128): m128 {.importc: "_mm_movehdup_ps", x86.}
    ## Duplicates high parts of the input
    ## Input:
    ##   { A0, A1, A2, A3 }
    ## Result:
    ##   { A1, A1, A3, A3 }
  func mm_moveldup_ps*(a: m128): m128 {.importc: "_mm_moveldup_ps", x86.}
    ## Duplicates low parts of the input
    ## Input:
    ##   { A0, A1, A2, A3 }
    ## Result:
    ##   { A0, A0, A2, A2 }

  # ############################################################
  #
  #                    SSE4.1 - integer - packed
  #
  # ############################################################

  func mm_mullo_epi32*(a, b: m128i): m128i {.importc: "_mm_mullo_epi32", x86.}
    ## Multiply element-wise 2 vectors of 4 32-bit ints
    ## into intermediate 4 64-bit ints, and keep the low 32-bit parts

  # ############################################################
  #
  #                    AVX - float32 - packed
  #
  # ############################################################

  func mm256_setzero_ps*(): m256 {.importc: "_mm256_setzero_ps", x86.}
  func mm256_set1_ps*(a: float32): m256 {.importc: "_mm256_set1_ps", x86.}
  func mm256_load_ps*(aligned_mem_addr: ptr float32): m256 {.importc: "_mm256_load_ps", x86.}
  func mm256_loadu_ps*(mem_addr: ptr float32): m256 {.importc: "_mm256_loadu_ps", x86.}
  func mm256_store_ps*(mem_addr: ptr float32, a: m256) {.importc: "_mm256_store_ps", x86.}
  func mm256_storeu_ps*(mem_addr: ptr float32, a: m256) {.importc: "_mm256_storeu_ps", x86.}
  func mm256_add_ps*(a, b: m256): m256 {.importc: "_mm256_add_ps", x86.}
  func mm256_mul_ps*(a, b: m256): m256 {.importc: "_mm256_mul_ps", x86.}
  func mm256_sub_ps*(a, b: m256): m256 {.importc: "_mm256_sub_ps", x86.}

  func mm256_and_ps*(a, b: m256): m256 {.importc: "_mm256_and_ps", x86.}
    ## Bitwise and
  func mm256_or_ps*(a, b: m256): m256 {.importc: "_mm256_or_ps", x86.}

  func mm256_min_ps*(a, b: m256): m256 {.importc: "_mm256_min_ps", x86.}
  func mm256_max_ps*(a, b: m256): m256 {.importc: "_mm256_max_ps", x86.}
  func mm256_castps256_ps128*(a: m256): m128 {.importc: "_mm256_castps256_ps128", x86.}
    ## Returns the lower part of a m256 in a m128
  func mm256_extractf128_ps*(v: m256, m: cint{lit}): m128 {.importc: "_mm256_extractf128_ps", x86.}
    ## Extracts the low part (m = 0) or high part (m = 1) of a m256 into a m128
    ## m must be a literal

  # ############################################################
  #
  #                   AVX - float64 - packed
  #
  # ############################################################

  func mm256_setzero_pd*(): m256d {.importc: "_mm256_setzero_pd", x86.}
  func mm256_set1_pd*(a: float64): m256d {.importc: "_mm256_set1_pd", x86.}
  func mm256_load_pd*(aligned_mem_addr: ptr float64): m256d {.importc: "_mm256_load_pd", x86.}
  func mm256_loadu_pd*(mem_addr: ptr float64): m256d {.importc: "_mm256_loadu_pd", x86.}
  func mm256_store_pd*(mem_addr: ptr float64, a: m256d) {.importc: "_mm256_store_pd", x86.}
  func mm256_storeu_pd*(mem_addr: ptr float64, a: m256d) {.importc: "_mm256_storeu_pd", x86.}
  func mm256_add_pd*(a, b: m256d): m256d {.importc: "_mm256_add_pd", x86.}
  func mm256_mul_pd*(a, b: m256d): m256d {.importc: "_mm256_mul_pd", x86.}

  # ############################################################
  #
  #                 AVX + FMA - float32/64 - packed
  #
  # ############################################################

  func mm256_fmadd_ps*(a, b, c: m256): m256 {.importc: "_mm256_fmadd_ps", x86.}
  func mm256_fmadd_pd*(a, b, c: m256d): m256d {.importc: "_mm256_fmadd_pd", x86.}

  # ############################################################
  #
  #                   AVX - integers - packed
  #
  # ############################################################

  func mm256_setzero_si256*(): m256i {.importc: "_mm256_setzero_si256", x86.}
  func mm256_set1_epi8*(a: int8 or uint8): m256i {.importc: "_mm256_set1_epi8", x86.}
  func mm256_set1_epi16*(a: int16 or uint16): m256i {.importc: "_mm256_set1_epi16", x86.}
  func mm256_set1_epi32*(a: int32 or uint32): m256i {.importc: "_mm256_set1_epi32", x86.}
  func mm256_set1_epi64x*(a: int64 or uint64): m256i {.importc: "_mm256_set1_epi64x", x86.}
  func mm256_load_si256*(mem_addr: ptr m256i): m256i {.importc: "_mm256_load_si256", x86.}
  func mm256_loadu_si256*(mem_addr: ptr m256i): m256i {.importc: "_mm256_loadu_si256", x86.}
  func mm256_storeu_si256*(mem_addr: ptr m256i, a: m256i) {.importc: "_mm256_storeu_si256", x86.}

  func mm256_castps_si256*(a: m256): m256i {.importc: "_mm256_castps_si256", x86.}
    ## Cast a float32x8 vectors into a 256-bit int vector with the same bit pattern
  func mm256_castsi256_ps*(a: m256i): m256 {.importc: "_mm256_castsi256_ps", x86.}
    ## Cast a 256-bit int vector into a float32x8 vector with the same bit pattern
  func mm256_cvtps_epi32*(a: m256): m256i {.importc: "_mm256_cvtps_epi32", x86.}
    ## Convert a float32x8 to int32x8
  func mm256_cvtepi32_ps*(a: m256i): m256 {.importc: "_mm256_cvtepi32_ps", x86.}
    ## Convert a int32x8 to float32x8

  # ############################################################
  #
  #                   AVX2 - integers - packed
  #
  # ############################################################

  func mm256_add_epi8*(a, b: m256i): m256i {.importc: "_mm256_add_epi8", x86.}
  func mm256_add_epi16*(a, b: m256i): m256i {.importc: "_mm256_add_epi16", x86.}
  func mm256_add_epi32*(a, b: m256i): m256i {.importc: "_mm256_add_epi32", x86.}
  func mm256_add_epi64*(a, b: m256i): m256i {.importc: "_mm256_add_epi64", x86.}

  func mm256_and_si256*(a, b: m256i): m256i {.importc: "_mm256_and_si256", x86.}
    ## Bitwise and
  func mm256_srli_epi64*(a: m256i, imm8: cint): m256i {.importc: "_mm256_srli_epi64", x86.}
    ## Logical right shift

  func mm256_mullo_epi16*(a, b: m256i): m256i {.importc: "_mm256_mullo_epi16", x86.}
    ## Multiply element-wise 2 vectors of 16 16-bit ints
    ## into intermediate 16 32-bit ints, and keep the low 16-bit parts

  func mm256_mullo_epi32*(a, b: m256i): m256i {.importc: "_mm256_mullo_epi32", x86.}
    ## Multiply element-wise 2 vectors of 8x 32-bit ints
    ## into intermediate 8x 64-bit ints, and keep the low 32-bit parts

  func mm256_shuffle_epi32*(a: m256i, imm8: cint): m256i {.importc: "_mm256_shuffle_epi32", x86.}
    ## Shuffle 32-bit integers in a according to the control in imm8
    ## Formula is in big endian representation
    ## a = {hi[a7, a6, a5, a4, lo[a3, a2, a1, a0]}
    ## dst = {d7, d6, d5, d4, d3, d2, d1, d0}
    ## imm8 = {bits76, bits54, bits32, bits10}
    ## d0 will refer a.lo[bits10]
    ## d1            a.lo[bits32]
    ## ...
    ## d4 will refer a.hi[bits10]
    ## d5            a.hi[bits32]

  func mm256_mul_epu32*(a: m256i, b: m256i): m256i {.importc: "_mm256_mul_epu32", x86.}
    ## From a = {a3_hi, a3_lo, a2_hi, a2_lo, a1_hi, a1_lo, a0_hi, a0_lo}
    ## with a3, a2, a1, a0 being 64-bit number
    ## and  b = {b3_hi, b3_lo, b2_hi, b2_lo, b1_hi, b1_lo, b0_hi, b0_lo}
    ##
    ## Result = {a3_lo * b3_lo, a2_lo * b2_lo, a1_lo * b1_lo, a0_lo * b0_lo}.
    ## This is an extended precision multiplication 32x32 -> 64

  func mm256_movemask_epi8*(a: m256i): int32 {.importc: "_mm256_movemask_epi8", x86.}
    ## Returns the most significant bit
    ## of each 8-bit elements in `a`

  func mm256_cmpgt_epi32*(a, b: m256i): m256i {.importc: "_mm256_cmpgt_epi32", x86.}
    ## Compare a greater than b

  func mm256_srli_epi32*(a: m256i, count: int32): m256i {.importc: "_mm256_srli_epi32", x86.}
  func mm256_slli_epi32*(a: m256i, count: int32): m256i {.importc: "_mm256_slli_epi32", x86.}

  func mm_i32gather_epi32*(m: ptr (uint32 or int32), i: m128i, s: int32): m128i {.importc: "_mm_i32gather_epi32", x86.}
  func mm256_i32gather_epi32*(m: ptr (uint32 or int32), i: m256i, s: int32): m256i {.importc: "_mm256_i32gather_epi32", x86.}

  # ############################################################
  #
  #                    AVX512 - float32 - packed
  #
  # ############################################################

  func mm512_setzero_ps*(): m512 {.importc: "_mm512_setzero_ps", x86.}
  func mm512_set1_ps*(a: float32): m512 {.importc: "_mm512_set1_ps", x86.}
  func mm512_load_ps*(aligned_mem_addr: ptr float32): m512 {.importc: "_mm512_load_ps", x86.}
  func mm512_loadu_ps*(mem_addr: ptr float32): m512 {.importc: "_mm512_loadu_ps", x86.}
  func mm512_store_ps*(mem_addr: ptr float32, a: m512) {.importc: "_mm512_store_ps", x86.}
  func mm512_storeu_ps*(mem_addr: ptr float32, a: m512) {.importc: "_mm512_storeu_ps", x86.}
  func mm512_add_ps*(a, b: m512): m512 {.importc: "_mm512_add_ps", x86.}
  func mm512_sub_ps*(a, b: m512): m512 {.importc: "_mm512_sub_ps", x86.}
  func mm512_mul_ps*(a, b: m512): m512 {.importc: "_mm512_mul_ps", x86.}
  func mm512_fmadd_ps*(a, b, c: m512): m512 {.importc: "_mm512_fmadd_ps", x86.}

  func mm512_min_ps*(a, b: m512): m512 {.importc: "_mm512_min_ps", x86.}
  func mm512_max_ps*(a, b: m512): m512 {.importc: "_mm512_max_ps", x86.}

  func mm512_or_ps*(a, b: m512): m512 {.importc: "_mm512_or_ps", x86.}

  # ############################################################
  #
  #                    AVX512 - float64 - packed
  #
  # ############################################################

  func mm512_setzero_pd*(): m512d {.importc: "_mm512_setzero_pd", x86.}
  func mm512_set1_pd*(a: float64): m512d {.importc: "_mm512_set1_pd", x86.}
  func mm512_load_pd*(aligned_mem_addr: ptr float64): m512d {.importc: "_mm512_load_pd", x86.}
  func mm512_loadu_pd*(mem_addr: ptr float64): m512d {.importc: "_mm512_loadu_pd", x86.}
  func mm512_store_pd*(mem_addr: ptr float64, a: m512d) {.importc: "_mm512_store_pd", x86.}
  func mm512_storeu_pd*(mem_addr: ptr float64, a: m512d) {.importc: "_mm512_storeu_pd", x86.}
  func mm512_add_pd*(a, b: m512d): m512d {.importc: "_mm512_add_pd", x86.}
  func mm512_mul_pd*(a, b: m512d): m512d {.importc: "_mm512_mul_pd", x86.}
  func mm512_fmadd_pd*(a, b, c: m512d): m512d {.importc: "_mm512_fmadd_pd", x86.}

  # # ############################################################
  # #
  # #                   AVX512 - integers - packed
  # #
  # # ############################################################

  func mm512_setzero_si512*(): m512i {.importc: "_mm512_setzero_si512", x86.}
  func mm512_set1_epi8*(a: int8 or uint8): m512i {.importc: "_mm512_set1_epi8", x86.}
  func mm512_set1_epi16*(a: int16 or uint16): m512i {.importc: "_mm512_set1_epi16", x86.}
  func mm512_set1_epi32*(a: int32 or uint32): m512i {.importc: "_mm512_set1_epi32", x86.}
  func mm512_set1_epi64*(a: int64 or uint64): m512i {.importc: "_mm512_set1_epi64", x86.}
  func mm512_load_si512*(mem_addr: ptr SomeInteger): m512i {.importc: "_mm512_load_si512", x86.}
  func mm512_loadu_si512*(mem_addr: ptr SomeInteger): m512i {.importc: "_mm512_loadu_si512", x86.}
  func mm512_storeu_si512*(mem_addr: ptr SomeInteger, a: m512i) {.importc: "_mm512_storeu_si512", x86.}

  func mm512_add_epi8*(a, b: m512i): m512i {.importc: "_mm512_add_epi8", x86.}
  func mm512_add_epi16*(a, b: m512i): m512i {.importc: "_mm512_add_epi16", x86.}
  func mm512_add_epi32*(a, b: m512i): m512i {.importc: "_mm512_add_epi32", x86.}
  func mm512_add_epi64*(a, b: m512i): m512i {.importc: "_mm512_add_epi64", x86.}

  func mm512_mullo_epi32*(a, b: m512i): m512i {.importc: "_mm512_mullo_epi32", x86.}
    ## Multiply element-wise 2 vectors of 16 32-bit ints
    ## into intermediate 16 32-bit ints, and keep the low 32-bit parts

  func mm512_mullo_epi64*(a, b: m512i): m512i {.importc: "_mm512_mullo_epi64", x86.}
    ## Multiply element-wise 2 vectors of 8x 64-bit ints
    ## into intermediate 8x 64-bit ints, and keep the low 64-bit parts

  func mm512_and_si512*(a, b: m512i): m512i {.importc: "_mm512_and_si512", x86.}
    ## Bitwise and

  func mm512_cmpgt_epi32_mask*(a, b: m512i): mmask16 {.importc: "_mm512_cmpgt_epi32_mask", x86.}
    ## Compare a greater than b, returns a 16-bit mask

  func mm512_maskz_set1_epi32*(k: mmask16, a: cint): m512i {.importc: "_mm512_maskz_set1_epi32", x86.}
    ## Compare a greater than b
    ## Broadcast 32-bit integer a to all elements of dst using zeromask k
    ## (elements are zeroed out when the corresponding mask bit is not set).

  func mm512_movm_epi32*(a: mmask16): m512i {.importc: "_mm512_movm_epi32", x86.}

  func mm512_movepi8_mask*(a: m512i): mmask64 {.importc: "_mm512_movepi8_mask", x86.}
    ## Returns the most significant bit
    ## of each 8-bit elements in `a`

  func mm512_srli_epi32*(a: m512i, count: int32): m512i {.importc: "_mm512_srli_epi32", x86.}
  func mm512_slli_epi32*(a: m512i, count: int32): m512i {.importc: "_mm512_slli_epi32", x86.}

  func mm512_i32gather_epi32*(i: m512i, m: ptr (uint32 or int32), s: int32): m512i {.importc: "_mm512_i32gather_epi32", x86.}
    ## Warning ⚠: Argument are switched compared to mm256_i32gather_epi32

  func mm512_castps_si512*(a: m512): m512i {.importc: "_mm512_castps_si512", x86.}
    ## Cast a float32x16 vectors into a 512-bit int vector with the same bit pattern
  func mm512_castsi512_ps*(a: m512i): m512 {.importc: "_mm512_castsi512_ps", x86.}
    ## Cast a 512-bit int vector into a float32x16 vector with the same bit pattern
  func mm512_cvtps_epi32*(a: m512): m512i {.importc: "_mm512_cvtps_epi32", x86.}
    ## Convert a float32x16 to int32x8
  func mm512_cvtepi32_ps*(a: m512i): m512 {.importc: "_mm512_cvtepi32_ps", x86.}
    ## Convert a int32x8 to float32x16

  func cvtmask64_u64*(a: mmask64): uint64 {.importc: "_cvtmask64_u64", x86.}
