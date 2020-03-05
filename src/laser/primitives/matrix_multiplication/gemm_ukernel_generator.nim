# Laser
# Copyright (c) 2018 Mamy André-Ratsimbazafy
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

import
  ../../compiler_optim_hints,
  ../../simd,
  ./gemm_tiling, ./gemm_utils,
  ./gemm_ukernel_generic,
  macros

# ############################################################
#
#             SIMD implementation generator
#
# ############################################################

# Macro ukernel_generator should be invoked in different files so that specific
# flags like "-mavx -mfma" are isolated.
# Add the corresponding compilation flags to "nim.cfg"

# #############################################################

template ukernel_simd_proc(ukernel_name, epilogue_name: NimNode, edge: bool) {.dirty.} =
  if edge:
    result.add quote do:
      proc `ukernel_name`*[ukernel: static MicroKernel](
            mr, nr, kc: int,
            alpha: `T`, packedA, packedB: ptr UncheckedArray[`T`],
            beta: `T`, vC: MatrixView[`T`]
          ) =

        let AB{.align_variable.} = ukernel_simd_impl(
          ukernel, `V`, packedA, packedB, kc,
          `simd_setZero`, `simd_load_aligned`, `simd_broadcast_value`, `simd_fma`
        )
        const
          # is_c_unit_stride = ukernel.extract_c_unit_stride()
          MR = ukernel.extract_mr()
          NR = ukernel.extract_nr()

        gebb_ukernel_edge_epilogue(
                alpha, to_ptr(AB, MR, NR, `T`),
                beta, vC, mr, nr
              )
  else:
    result.add quote do:
      proc `ukernel_name`*[ukernel: static MicroKernel](
            kc: int,
            alpha: `T`, packedA, packedB: ptr UncheckedArray[`T`],
            beta: `T`, vC: MatrixView[`T`]
          ) =
        let AB{.align_variable.} = ukernel_simd_impl(
          ukernel, `V`, packedA, packedB, kc,
          `simd_setZero`, `simd_load_aligned`, `simd_broadcast_value`, `simd_fma`
        )
        const
          # is_c_unit_stride = ukernel.extract_c_unit_stride()
          MR = ukernel.extract_mr()
          NR = ukernel.extract_nr()

        # when is_c_unit_stride:
        #   `epilogue_name`(alpha, AB, beta, vC)
        # else:
        gebb_ukernel_epilogue_fallback(
          alpha, to_ptr(AB, MR, NR, `T`),
          beta, vC)

# #############################################################

template epilogue() {.dirty.} =
  result.add quote do:
    proc `epilogue_name`[MR, NbVecs: static int](
            alpha: `T`, AB: array[MR, array[NbVecs, `V`]],
            beta: `T`, vC: MatrixView[`T`]
          ) =
      template C(i,j: int): untyped {.dirty.} =
        vC.buffer[i*vC.rowStride + j*`nb_scalars`]

      if beta == 0.`T`:
        for i in 0 ..< MR:
          for j in 0 ..< NbVecs:
            `simd_store_unaligned`(C(i,j).addr, `simd_setZero`())
      elif beta != 1.`T`:
        let beta_vec = `simd_broadcast_value`(beta)
        for i in 0 ..< MR:
          for j in 0 ..< NbVecs:
            `simd_store_unaligned`(C(i,j).addr, `simd_mul`(beta_vec, C(i,j).addr.`simd_load_unaligned`))

      if alpha == 1.`T`:
        for i in 0 ..< MR:
          for j in 0 ..< NbVecs:
            `simd_store_unaligned`(C(i,j).addr, `simd_add`(AB[i][j], C(i,j).addr.`simd_load_unaligned`))
      else:
        let alpha_vec = `simd_broadcast_value`(alpha)
        for i in 0 ..< MR:
          for j in 0 ..< NbVecs:
            `simd_store_unaligned`(C(i,j).addr, `simd_fma`(alpha_vec, AB[i][j], C(i,j).addr.`simd_load_unaligned`))

# #############################################################

macro ukernel_generator*(
      simd: static CPUFeatureX86,
      typ: untyped,
      vectype: untyped,
      nb_scalars: static int,
      simd_setZero: untyped,
      simd_broadcast_value: untyped,
      simd_load_aligned: untyped,
      simd_load_unaligned: untyped,
      simd_store_unaligned: untyped,
      simd_mul: untyped,
      simd_add: untyped,
      simd_fma: untyped,
    ): untyped =

  let T = newIdentNode($typ)
  let V = newIdentNode($vectype)
  let epilogue_name = newIdentNode("gebb_ukernel_epilogue_" & $T & "_" & $simd)
  result = newStmtList()

  # 1. Generate the epilogue function
  epilogue()

  # 2. Generate the microkernels for the general and edge cases
  block:
    let ukernel_name = newIdentNode("gebb_ukernel_" & $T & "_" & $simd)
    ukernel_simd_proc(ukernel_name, epilogue_name, edge = false)
  block:
    let ukernel_name = newIdentNode("gebb_ukernel_edge_" & $T & "_" & $simd)
    ukernel_simd_proc(ukernel_name, epilogue_name, edge = true)

# ############################################################
#
#             Actual SIMD implementation
#
# ############################################################

macro ukernel_simd_impl*(
      ukernel: static MicroKernel, V: untyped, A, B: untyped, kc: int,
      simd_setZero, simd_load_aligned, simd_broadcast_value, simd_fma: untyped
    ): untyped =


  let MR = ukernel.mr
  let NR = ukernel.nr

  if false: # Debug implementation
    result = quote do:
      var AB{.align_variable.}: array[`MR`, array[`NR`, float64]]
      var  A {.restrict.} = assume_aligned packedA # [kc, mc] by chunks of mr
      var  B {.restrict.} = assume_aligned packedB # [kc, nc] by chunks of nr

      for k in 0 ..< kc:
        prefetch(B[(k+1)*`NR`].addr, Read, LowTemporalLocality)
        for i in 0 ..< `MR`:
          for j in 0 ..< `NR`-1:
            AB[i][j] += A[k*`MR`+i] * B[k*`NR`+j]
      AB

  else: # Vectorized implementation
    result = newStmtList()

    ## ukernel config
    let
      MR = ukernel.mr
      NR = ukernel.nr
      NbVecs = ukernel.nb_vecs_nr # == NR div NbScalars
      NbScalars = ukernel.nb_scalars

    ## Registers
    # We keep all C in registers MR*NR size occupying MR*NbVecs
    # We keep NbVecs slivers of A and B for C updates
    var
      rA: seq[NimNode]           # array[NbVecs, V]
      rB: seq[NimNode]           # array[NbVecs, V]
      rAB = nnkBracket.newTree() # array[MR, array[NbVecs, V]]
    for jj in 0 ..< NbVecs:
      rA.add genSym(nskVar, "A" & $jj)
      rB.add genSym(nskVar, "B" & $jj)
    for i in 0 ..< MR:
      var rABi = nnkBracket.newTree()
      for j in 0 ..< NbVecs:
        rABi.add genSym(nskVar, "AB" & $i & "__" & $j)
      rAB.add rABi

    ## Declare
    var declBody = newStmtList()
    for a in rA:
      declBody.add quote do:
        var `a`{.noinit.}: `V`
    for b in rB:
      declBody.add quote do:
        var `b`{.noinit.}: `V`
    for i in 0 ..< MR:
      for j in 0 ..< NbVecs:
        let ab = rAB[i][j]
        declBody.add quote do:
          var `ab` = `simd_setZero`()

    let k = genSym(nskForVar)

    ## Prefetch
    var prefetchBody = newStmtList()
    for jj in 0 ..< NbVecs:
      prefetchBody.add quote do:
        prefetch(`B`[(`k`+1)*`NR`+`jj`*`NbScalars`].addr, Read, LowTemporalLocality)

    ## Load
    var loadBody = newStmtList()
    for jj in 0 ..< NbVecs:
      let b = rB[jj]
      loadBody.add quote do:
        `b` = `simd_load_aligned`(`B`[`k`*`NR`+`jj`*`NbScalars`].addr)

    ## Interleaved broadcast and FMA
    var bcast_fma = newStmtList()
    block:
      let a0 = rA[0]
      bcast_fma.add quote do:
        `a0` = `simd_broadcast_value`(`A`[`k`*`MR`])

    for i in 0 ..< MR:
      # broadcast next iteration
      let next_register_idx = (i+1) mod NbVecs
      let a_next = rA[next_register_idx]
      bcast_fma.add quote do:
        # At the edge: `i`+1 = MR so equivalent to loading A[(k+1)*MR]
        `a_next` = `simd_broadcast_value`(`A`[`k`*`MR`+(`i`+1)])

      # load current
      let a = rA[i mod NbVecs]

      # Do FMA on the current one
      for jj in 0 ..< NbVecs:
        let b = rB[jj]
        let AB = rAB[i][jj]
        bcast_fma.add quote do:
          `AB` = `simd_fma`(`a`, `b`, `AB`)

    ## Assemble:
    result = quote do:
      `declBody`
      for `k` in 0 ..< `kc`:
        `loadBody`
        `prefetchBody`
        `bcast_fma`
      ## Write registers to a MR/NR array
      `rAB`
