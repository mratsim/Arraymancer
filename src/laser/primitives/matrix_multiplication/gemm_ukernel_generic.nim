# Laser
# Copyright (c) 2018 Mamy André-Ratsimbazafy
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

# Generic microkernel for matrix multiplication

import
  # ../../cpuinfo,
  ../../compiler_optim_hints,
  ./gemm_tiling, ./gemm_utils,
  macros

withCompilerOptimHints()

# ############################################################
#
#          Generic GEBB microkernel implementation
#
# ############################################################

template ukernel_generic_impl*(){.dirty.} =
  const
    MR = ukernel.extract_mr()
    NR = ukernel.extract_nr()

  var AB{.align_variable.}: array[MR, array[NR, T]]
  var  A {.restrict.} = assume_aligned packedA # [kc, mc] by chunks of mr
  var  B {.restrict.} = assume_aligned packedB # [kc, nc] by chunks of nr

  for k in 0 ..< kc:
    prefetch(B[(k+1)*NR].addr, Read, LowTemporalLocality)
    for i in 0 ..< MR:
      for j in `||`(0, NR-1, "simd"):
        AB[i][j] += A[k*MR+i] * B[k*NR+j]

# ############################################################
#
#          Fallback Generic version
#
# ############################################################
#
# Cases
# 1. C *=   β, starting default
# 2. C  =  AB, if β = 0 and α = 1
# 3. C  = αAB, if β = 0 and α = 1
# 4. C +=  AB, if α = 1
# 5. C += αAB, if α = 1
#
# TODO: Fused operations like relu/sigmoid/tanh
#       should be done here as well

proc gebb_ukernel_epilogue_fallback*[MR, NR: static int, T](
      alpha: T, AB: ptr array[MR, array[NR, T]],
      beta: T,  vC: MatrixView[T]
    ){.inline.} =

  let pAB{.restrict.} = assume_aligned cast[ptr array[MR, array[NR, T]]](AB[0][0].unsafeAddr)

  if beta == 0.T:
    for i in 0 ..< MR:
      for j in 0 ..< NR:
        vC[i, j] = 0.T
  elif beta != 1.T:                  # C *= β
    for i in 0 ..< MR:
      for j in 0 ..< NR:
        vC[i, j] *= beta

  if alpha == 1.T:                   # C += AB
    for i in 0 ..< MR:
      for j in 0 ..< NR:
        vC[i, j] += pAB[i][j]
  else:                              # C += αAB
    for i in 0 ..< MR:
      for j in 0 ..< NR:
        vC[i, j] += alpha * pAB[i][j]

  # TODO: Fused operations like relu/sigmoid/tanh
  #       should be done here as well

proc gebb_ukernel_fallback*[T; ukernel: static MicroKernel](
      kc: int,
      alpha: T, packedA, packedB: ptr UncheckedArray[T],
      beta: T, vC: MatrixView[T]
    ) =
  ukernel_generic_impl()

  gebb_ukernel_epilogue_fallback(alpha, to_ptr(AB, MR, NR, T), beta, vC)

# ############################################################
#
#                      Matrix edges
#
# ############################################################

func gebb_ukernel_edge_epilogue*[MR, NR: static int, T](
      alpha: T, AB: ptr array[MR, array[NR, T]],
      beta: T,  vC: MatrixView[T],
      mr, nr: int # Tail to process
    ){.inline.} =

  let pAB{.restrict.} = assume_aligned cast[ptr array[MR, array[NR, T]]](AB[0][0].unsafeAddr)

  if beta == 0.T:
    if alpha == 1.T:                   # C = AB
      for i in 0 ..< mr:
        for j in 0 ..< nr:
          vC[i, j] = pAB[i][j]
    else:                              # C = αAB
      for i in 0 ..< mr:
        for j in 0 ..< nr:
          vC[i, j] = alpha * pAB[i][j]
  else:                                # C *= β
    for i in 0 ..< mr:
      for j in 0 ..< nr:
        vC[i, j] *= beta

    if alpha == 1.T:                   # C += AB
      for i in 0 ..< mr:
        for j in 0 ..< nr:
          vC[i, j] += pAB[i][j]
    else:                              # C += αAB
      for i in 0 ..< mr:
        for j in 0 ..< nr:
          vC[i, j] += alpha * pAB[i][j]

  # TODO: Fused operations like relu/sigmoid/tanh
  #       should be done here as well

proc gebb_ukernel_edge_fallback*[T; ukernel: static MicroKernel](
      mr, nr, kc: int,
      alpha: T, packedA, packedB: ptr UncheckedArray[T],
      beta: T, vC: MatrixView[T]
    ) =
  ukernel_generic_impl()
  gebb_ukernel_edge_epilogue(alpha, to_ptr(AB, MR, NR, T), beta, vC, mr, nr)
