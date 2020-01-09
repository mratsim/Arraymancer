# Laser
# Copyright (c) 2018 Mamy André-Ratsimbazafy
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

import
  macros,
  ../../compiler_optim_hints,
  ./gemm_tiling, ./gemm_utils,
  ./gemm_ukernel_generic,
  ./gemm_ukernel_sse,
  ./gemm_ukernel_sse2,
  ./gemm_ukernel_sse4_1,
  ./gemm_ukernel_avx,
  ./gemm_ukernel_avx_fma,
  ./gemm_ukernel_avx2,
  ./gemm_ukernel_avx512

{.experimental: "dynamicBindSym".}

# ############################################################
#
#            Dispatch with runtime cpu detection
#
# ############################################################

template dispatch_common {.dirty.} =
  let simd = ukernel.cpu_simd
  let MR = ukernel.mr
  let nb_scalars = ukernel.nb_scalars

  result = newStmtList()

  # 1. Prefetch packedB (used in microkernel)
  #         and C (used in epilogue update)
  result.add quote do:
    prefetch(`packedB`, Read, LowTemporalLocality)
    prefetch(`packedB` + `nb_scalars`, Read, LowTemporalLocality)
    for i in 0 ..< `MR`:
      prefetch(`vC`[i, 0].addr, Write, HighTemporalLocality)

  # 2. Dispatch according to type and SIMD support
  let symT = getTypeInst(alpha)

macro dispatch_general(
    ukernel: static MicroKernel,
    kc: int,
    alpha: typed, packedA, packedB: ptr UncheckedArray[typed],
    beta: typed, vC: MatrixView[typed]
  ): untyped =

  dispatch_common()

  # 2.1. No SIMD case
  if simd == x86_Generic:
    result.add quote do:
      gebb_ukernel_fallback[`symT`, ukernel]( # Hack: ukernel is generic from the calling proc
              `kc`,
        `alpha`, `packedA`, `packedB`,
        `beta`, `vC`
      )
    return

  # 2.2. SIMD case
  let simdTag = $simd
  let ukernel_name = bindSym("gebb_ukernel_" & $symT & "_" & simdTag)
  result.add quote do:
    `ukernel_name`[ukernel]( # Hack: ukernel is generic from the calling proc
      `kc`,
      `alpha`, `packedA`, `packedB`,
      `beta`, `vC`
    )

proc gebb_ukernel*[T; ukernel: static MicroKernel](
      kc: int,
      alpha: T, packedA, packedB: ptr UncheckedArray[T],
      beta: T, vC: MatrixView[T]
    ){.inline.} =

  ukernel.dispatch_general(kc, alpha, packedA, packedB, beta, vC)


# ############################################################
#
#                      Exported proc
#
# ############################################################

macro dispatch_edge(
    ukernel: static MicroKernel,
    mr, nr, kc: int,
    alpha: typed, packedA, packedB: ptr UncheckedArray[typed],
    beta: typed, vC: MatrixView[typed]
  ): untyped =

  dispatch_common()

  # 2.1. No SIMD case
  if simd == x86_Generic:
    result.add quote do:
      gebb_ukernel_edge_fallback[`symT`, ukernel]( # Hack: ukernel is generic from the calling proc
        `mr`, `nr`, `kc`,
        `alpha`, `packedA`, `packedB`,
        `beta`, `vC`
      )
    return

  # 2.2. SIMD case
  let simdTag = $simd
  let ukernel_name = bindSym("gebb_ukernel_edge_" & $symT & "_" & simdTag)
  result.add quote do:
    `ukernel_name`[ukernel]( # Hack: ukernel is generic from the calling proc
      `mr`, `nr`, `kc`,
      `alpha`, `packedA`, `packedB`,
      `beta`, `vC`
    )

proc gebb_ukernel_edge*[T; ukernel: static MicroKernel](
      mr, nr, kc: int,
      alpha: T, packedA, packedB: ptr UncheckedArray[T],
      beta: T, vC: MatrixView[T]
    ){.inline.} =

  ukernel.dispatch_edge(mr, nr, kc, alpha, packedA, packedB, beta, vC)
