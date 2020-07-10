# Laser
# Copyright (c) 2018 Mamy André-Ratsimbazafy
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

import
  # ../../cpuinfo,
  ../../compiler_optim_hints, ../../openmp,
  ../../private/align_unroller,
  ./gemm_tiling, ./gemm_utils, ./gemm_packing,
  ./gemm_ukernel_dispatch, ./gemm

withCompilerOptimHints()

# ############################################################
#
#            GEMM Prepacked Matrices A and B
#
# ############################################################

template dispatch(
    return_void: static bool,
    func_call: untyped): untyped {.dirty.} =
  ## Warning: statements after dispatch are unreachable
  template dispatch_opt(cpu_features: static CPUFeatureX86): untyped {.dirty.} =
    ## Dispatch depending on detected CPU features.
    type A = T # workaround "Cannot evaluate at compile-time
    # c_unit_stride is not relevant here
    const ukernel = cpu_features.x86_ukernel(A, c_unit_stride = false)

    when return_void:
      func_call
      return
    else:
      return func_call

  when defined(i386) or defined(amd64):
    when T is float32:
      if cpuinfo_has_x86_avx512f():   dispatch_opt(x86_AVX512)
      elif cpuinfo_has_x86_fma3():    dispatch_opt(x86_AVX_FMA)
      elif cpuinfo_has_x86_avx():     dispatch_opt(x86_AVX)
      elif cpuinfo_has_x86_sse():     dispatch_opt(x86_SSE)
    elif T is float64:
      if cpuinfo_has_x86_avx512f():   dispatch_opt(x86_AVX512)
      elif cpuinfo_has_x86_fma3():    dispatch_opt(x86_AVX_FMA)
      elif cpuinfo_has_x86_avx():     dispatch_opt(x86_AVX)
      elif cpuinfo_has_x86_sse2():    dispatch_opt(x86_SSE2)
    elif T is int32 or T is uint32:
      if cpuinfo_has_x86_avx512f():   dispatch_opt(x86_AVX512)
      elif cpuinfo_has_x86_avx2():    dispatch_opt(x86_AVX2)
      elif cpuinfo_has_x86_sse41():   dispatch_opt(x86_SSE4_1)
      elif cpuinfo_has_x86_sse2():    dispatch_opt(x86_SSE2)
    elif T is int64:
      if cpuinfo_has_x86_avx512f():   dispatch_opt(x86_AVX512)
      elif cpuinfo_has_x86_sse2():    dispatch_opt(x86_SSE2)
  dispatch_opt(x86_Generic)

# ############################################################
#
#                    Packing B
#
# ############################################################

func gemm_prepackB_mem_required_impl*(
  ukernel: static MicroKernel,
  T: typedesc,
  M, N, K: int): int =

  let (MC, NC, KC) = ukernel.partitionMNK(T, M, N, K)
  const NR = ukernel.nr

  let pc_num_iter = get_num_tiles(K, KC)
  let upanelB_size = KC * round_step_up(NC, NR)

  result = T.sizeof * upanelB_size * pc_num_iter

func gemm_prepackB_mem_required*(
  T: type,
  M, N, K: int): int =
  ## Returns the amount of memory that needs to be preallocated
  ## to pack matrix B.

  dispatch(return_void = false):
    gemm_prepackB_mem_required_impl(
      ukernel, T, M, N, K
    )

proc gemm_prepackB_impl[T; ukernel: static MicroKernel](
        dst: ptr UncheckedArray[T],
        M, N, K: int,
        vB: MatrixView[T]
      ) =

  let (MC, NC, KC) = ukernel.partitionMNK(T, M, N, K)
  let pc_num_iter = get_num_tiles(K, KC)
  let upanelB_size = KC * round_step_up(NC, ukernel.nr)
  for pcb in 0||(pc_num_iter-1):
    let packB = dst + pcb * upanelB_size
    prefetch(packB, Write, LowTemporalLocality)

    let pc = pcb * KC
    let kc = min(K - pc, KC)
    let kcncB = vB.stride(pc, 0)

    # Note: pack_B also creates a parallel region
    #       this will cause issues if omp_get_nested = 1
    pack_B_kc_nc[T, ukernel](
      packB,
      kc, NC, kcncB
    )

proc gemm_prepackB*[T](
        dst_packedB: ptr (T or UncheckedArray[T]),
        M, N, K: int,
        src_B: ptr T, rowStrideB, colStrideB: int) =
  ## Prepack matrix B of shape KxN
  ## and strides rowStrideB and colStrideB
  ## for matrix multiplication.
  ## B must be 64-bit aligned.
  ##
  ## For optimal performance packing is machine and architecture dependent
  ## i.e. it depends on detected features like AVX and number of cores
  ## and may depend on your machine cache sizes in the future.
  ## It is unsafe to store or serialize it.

  doAssert (cast[int](dst_packedB) and 63) == 0, "The destination pointer must be 64-bit aligned"

  let vB = src_B.toMatrixView(rowStrideB, colStrideB)
  let dst = cast[ptr UncheckedArray[T]](dst_packedB)

  dispatch(return_void = true):
    gemm_prepackB_impl[T, ukernel](
      dst,
      M, N, K,
      vB
    )

# ############################################################
#
#                    Packing A
#
# ############################################################

func gemm_prepackA_mem_required_impl*(
  ukernel: static MicroKernel,
  T: typedesc,
  M, N, K: int): int =

  let (MC, NC, KC) = ukernel.partitionMNK(T, M, N, K)
  const MR = ukernel.mr

  let pc_num_iter = get_num_tiles(K, KC)
  let ic_num_iter = get_num_tiles(M, MC)
  let upanelA_size = KC * round_step_up(MC, MR)

  result = T.sizeof * upanelA_size * pc_num_iter * ic_num_iter

func gemm_prepackA_mem_required*(
  T: typedesc,
  M, N, K: int): int =
  ## Returns the amount of memory that needs to be preallocated
  ## to pack matrix B.

  dispatch(return_void = false):
    gemm_prepackA_mem_required_impl(
      ukernel, T, M, N, K
    )

proc gemm_prepackA_impl[T; ukernel: static MicroKernel](
        dst: ptr UncheckedArray[T],
        M, N, K: int,
        vA: MatrixView[T]
      ) =

  let (MC, NC, KC) = ukernel.partitionMNK(T, M, N, K)
  const MR = ukernel.mr

  let pc_num_iter = get_num_tiles(K, KC)
  let ic_num_iter = get_num_tiles(M, MC)
  let upanelA_size = KC * round_step_up(MC, MR)

  for pcb in 0||(pc_num_iter-1):
    let pc = pcb * KC
    let kc = min(K - pc, KC)

    for icb in 0 ..< ic_num_iter:
      let packA = dst + pc*pc_num_iter + icb*upanelA_size
      prefetch(packA, Write, LowTemporalLocality)
      let ic = icb * MC
      let mc = min(M-ic, MC)

      let mckcA = vA.stride(ic, pc)
      pack_A_mc_kc[T, ukernel](packA, mc, kc, mckcA)

proc gemm_prepackA*[T](
        dst_packedA: ptr (T or UncheckedArray[T]),
        M, N, K: int,
        src_A: ptr T, rowStrideA, colStrideA: int) =
  ## Prepack matrix A of shape MxK
  ## and strides rowStrideA and colStrideA
  ## for matrix multiplication.
  ## A must be 64-bit aligned.
  ##
  ## For optimal performance packing is machine and architecture dependent
  ## i.e. it depends on detected features like AVX and number of cores
  ## and may depend on your machine cache sizes in the future.
  ## It is unsafe to store or serialize it.

  doAssert (cast[int](dst_packedA) and 63) == 0, "The destination pointer must be 64-bit aligned"

  let vA = src_A.toMatrixView(rowStrideA, colStrideA)
  let dst = cast[ptr UncheckedArray[T]](dst_packedA)

  dispatch(return_void = true):
    gemm_prepackA_impl[T, ukernel](
      dst,
      M, N, K,
      vA
    )

# ############################################################
#
#                    Prepacked GEMM
#
# ############################################################

proc gemm_packed_impl[T](
      ukernel: static MicroKernel,
      M, N, K: int,
      alpha: T, packedA, packedB: ptr (T or UncheckedArray[T]),
      beta: T, vC: MatrixView[T]
    ) =

  withCompilerOptimHints()

  const
    MR = ukernel.mr
    NR = ukernel.nr
    PT = ukernel.pt

  let
    parallelize = M*N*K > PT*PT*PT

    (MC, NC, KC) = ukernel.partitionMNK(T, M, N, K)
    pc_num_iter = get_num_tiles(K, KC)
    ic_num_iter = get_num_tiles(M, MC)

    upanelB_size = KC * round_step_up(NC, NR)
    upanelA_size = KC * round_step_up(MC, MR)


  # ######################################
  # 2.   for pc = 0,...,k−1 in steps of kc
  for pcb in 0 ..< pc_num_iter:
    let packedB{.restrict.} = cast[ptr UncheckedArray[T]](packedB + pcb * upanelB_size)
    let pc = pcb * KC
    let kc = min(K - pc, KC)

    # First time writing to C, we scale it, otherwise accumulate
    let beta = if pc == 0: beta else: 1.T

    omp_parallel_if(parallelize):
      # ####################################
      # 3. for ic = 0,...,m−1 in steps of mc
      omp_for(icb, ic_num_iter, use_simd=false, nowait=true):
        let packedA{.restrict.} = cast[ptr UncheckedArray[T]](packedA + icb * upanelA_size)
        let ic = icb * MC
        let mc = min(M-ic, MC)

        gebp_mkernel[T, ukernel](
          mc, NC, kc,
          alpha, packedA, packedB,
          beta, vc.stride(ic, 0)
        )

proc gemm_packed*[T: SomeNumber](
      M, N, K: int,
      alpha: T,
      packedA: ptr (T or UncheckedArray[T]),
      packedB: ptr (T or UncheckedArray[T]),
      beta: T,
      C: ptr (T or UncheckedArray[T]),
      rowStrideC, colStrideC: int) =

  let vC = C.toMatrixView(rowStrideC, colStrideC)

  dispatch(return_void = true):
    # TODO - dispatch specialization when C is unit strided
    ukernel.gemm_packed_impl(
      M, N, K,
      alpha, packedA, packedB,
      beta, vC
    )

# ############################################################
#
#                       Private tests
#
# ############################################################

when false:
  ## these tests don't work in arraymancer, since the imported files are not
  ## part of arraymancer's repository.
  import
    ../../tensor/[allocator, datatypes, initialization],
    strformat

  proc toPtr*[T](t: Tensor[T]): ptr T =
    cast[ptr T](t.unsafe_raw_data)

  proc `$`[T](t: Tensor[T]): string =
    var tmp = newSeq[T](t.size)
    copyMem(tmp[0].addr, cast[ptr T](t.unsafe_raw_data), t.size * sizeof(T))
    result = $tmp

  proc pack_and_test[M, N, K: static int; T](
          a: array[M, array[K, T]],
          b: array[K, array[N, T]],
          ab: array[M, array[N, T]]
        ) =
    echo "M: ", M
    echo "N: ", N
    echo "K: ", K
    echo fmt"A [{M}x{K}] * B[{K}x{N}] -> C[{M}x{N}]"
    let packedA_size = gemm_prepackA_mem_required(T, M, N, K)
    var packA = newTensor[T](packedA_size)
    gemm_prepackA(
      packA.toPtr,
      M, N, K,
      a[0][0].unsafeAddr,
      K, 1
    )
    # echo packA

    let packedB_size = gemm_prepackB_mem_required(T, M, N, K)
    var packB = newTensor[T](packedB_size)
    gemm_prepackB(
      packB.toPtr,
      M, N, K,
      b[0][0].unsafeAddr,
      N, 1
    )
    # echo packB

    var res_ab: array[M, array[N, T]]
    gemm_packed(
      M, N, K,
      T(1), packA.toPtr, packB.toPtr,
      T(0), res_ab[0][0].addr, N, 1
    )

    doAssert res_ab == ab, $res_ab
    echo "SUCCESS\n"

  # Tests
  block:
    let a = [[1.0, 2, 3],
             [4.0, 5, 6],
             [7.0, 8, 9]]

    let b = [[1.0, 2, 3],
             [4.0, 5, 6],
             [7.0, 8, 9]]

    let ab = [[30.0, 36, 42],
             [66.0, 81, 96],
             [102.0, 126, 150]]

    pack_and_test(a, b, ab)

  block:
    let a = [[1.0, 2, 3],
             [1.0, 1, 1],
             [1.0, 1, 1]]

    let b = [[1.0, 1],
             [1.0, 1],
             [1.0, 1]]

    let ab = [[6.0, 6],
              [3.0, 3],
              [3.0, 3]]

    pack_and_test(a, b, ab)

  block:
    let a = [[1.0, 2, 3],
             [4.0, 5, 6],
             [7.0, 8, 9]]

    let b = [[1.0, 1],
             [1.0, 1],
             [1.0, 1]]

    let ab = [[ 6.0,  6],
              [15.0, 15],
              [24.0, 24]]

    pack_and_test(a, b, ab)

  block:
    let a = [[1.0,2,3],
             [4.0,5,6]]

    let b = [[7.0,  8],
             [9.0, 10],
             [11.0,12]]

    let ab = [[ 58.0, 64],
              [139.0,154]]

    pack_and_test(a, b, ab)

  block:
    # example from http://www.intmath.com/matrices-determinants/matrix-multiplication-examples.php
    echo "\n## (M x K) * (K x N) with M < N"
    let a = [[-2,-3,-1],
             [ 3, 0, 4]]
    let b = [[ 1, 5, 2,-1],
             [-3, 0, 3, 4],
             [ 6,-2, 7,-4]]

    let ab = [[ 1,-8,-20, -6],
              [27, 7, 34,-19]]

    pack_and_test(a, b, ab)

  block:
    # from http://www.calcul.com/show/calculator/matrix-multiplication_;5;5;5;5?matrix1=[[%225%22,%226%22,%225%22,%228%22],[%228%22,%222%22,%228%22,%228%22],[%220%22,%225%22,%224%22,%220%22],[%224%22,%220%22,%225%22,%226%22],[%224%22,%225%22,%220%22,%223%22]]&matrix2=[[%225%22,%223%22,%226%22,%220%22],[%225%22,%222%22,%223%22,%223%22],[%228%22,%228%22,%222%22,%220%22],[%227%22,%227%22,%220%22,%220%22]]&operator=*
    echo "\n## (M x K) * (K x N) with M > N and M > block-size (4x4)"
    let a =  [[5,6,5,8],
              [8,2,8,8],
              [0,5,4,0],
              [4,0,5,6],
              [4,5,0,3]]
    let b =  [[5,3,6,0],
              [5,2,3,3],
              [8,8,2,0],
              [7,7,0,0]]

    let ab = [[151,123,58,18],
              [170,148,70, 6],
              [ 57, 42,23,15],
              [102, 94,34, 0],
              [ 66, 43,39,15]]

    pack_and_test(a, b, ab)

  block:
    let a =  [[2, 4,  3,  1,  3,  1,  3,  1],
              [4, 3,  2,  4,  1,  0,  0,  0]]


    let b =  [[2, 2],
              [2, 1],
              [0, 3],
              [0, 1],
              [0, 2],
              [4, 3],
              [3, 3],
              [2, 1]]

    let ab = [[27,37],
              [14,23]]

    pack_and_test(a, b, ab)

  block:
    let a =  [[2, 1],
              [1, 3],
              [2, 1],
              [1, 0],
              [3, 4],
              [2, 4],
              [3, 1],
              [4, 0]]


    let b =  [[2, 2,  0,  4,  0,  0,  4,  2],
              [2, 1,  2,  1,  2,  4,  4,  1]]

    let ab = [[ 6,  5,  2,  9,  2,  4, 12,  5],
              [ 8,  5,  6,  7,  6, 12, 16,  5],
              [ 6,  5,  2,  9,  2,  4, 12,  5],
              [ 2,  2,  0,  4,  0,  0,  4,  2],
              [14, 10,  8, 16,  8, 16, 28, 10],
              [12,  8,  8, 12,  8, 16, 24,  8],
              [ 8,  7,  2, 13,  2,  4, 16,  7],
              [ 8,  8,  0, 16,  0,  0, 16,  8]]

    pack_and_test(a, b, ab)

  block:
    # from http://www.calcul.com/show/calculator/matrix-multiplication?matrix1=[[%222%22,%224%22,%223%22,%221%22,%223%22,%221%22,%223%22,%221%22],[%221%22,%222%22,%221%22,%221%22,%222%22,%220%22,%224%22,%223%22],[%222%22,%220%22,%220%22,%223%22,%220%22,%224%22,%224%22,%221%22],[%221%22,%221%22,%224%22,%220%22,%223%22,%221%22,%223%22,%220%22],[%223%22,%224%22,%221%22,%221%22,%224%22,%222%22,%223%22,%224%22],[%222%22,%224%22,%220%22,%222%22,%223%22,%223%22,%223%22,%224%22],[%223%22,%220%22,%220%22,%223%22,%221%22,%224%22,%223%22,%221%22],[%224%22,%223%22,%222%22,%224%22,%221%22,%220%22,%220%22,%220%22]]&matrix2=[[%222%22,%222%22,%220%22,%224%22,%220%22,%220%22,%224%22,%222%22],[%222%22,%220%22,%220%22,%221%22,%221%22,%221%22,%223%22,%221%22],[%220%22,%222%22,%222%22,%220%22,%222%22,%222%22,%223%22,%223%22],[%220%22,%220%22,%221%22,%220%22,%224%22,%222%22,%224%22,%221%22],[%220%22,%220%22,%221%22,%223%22,%224%22,%222%22,%224%22,%222%22],[%224%22,%223%22,%224%22,%221%22,%224%22,%224%22,%220%22,%223%22],[%223%22,%223%22,%220%22,%222%22,%221%22,%222%22,%223%22,%223%22],[%222%22,%221%22,%222%22,%221%22,%222%22,%224%22,%224%22,%221%22]]&operator=*
    echo "\n## (N x N) * (N x N) with N multiple of block size"

    let a =  [[2, 4,  3,  1,  3,  1,  3,  1],
              [1, 2,  1,  1,  2,  0,  4,  3],
              [2, 0,  0,  3,  0,  4,  4,  1],
              [1, 1,  4,  0,  3,  1,  3,  0],
              [3, 4,  1,  1,  4,  2,  3,  4],
              [2, 4,  0,  2,  3,  3,  3,  4],
              [3, 0,  0,  3,  1,  4,  3,  1],
              [4, 3,  2,  4,  1,  0,  0,  0]]


    let b =  [[2, 2,  0,  4,  0,  0,  4,  2],
              [2, 0,  0,  1,  1,  1,  3,  1],
              [0, 2,  2,  0,  2,  2,  3,  3],
              [0, 0,  1,  0,  4,  2,  4,  1],
              [0, 0,  1,  3,  4,  2,  4,  2],
              [4, 3,  4,  1,  4,  4,  0,  3],
              [3, 3,  0,  2,  1,  2,  3,  3],
              [2, 1,  2,  1,  2,  4,  4,  1]]

    let ab = [[27,23,16,29,35,32,58,37],
              [24,19,11,23,26,30,49,27],
              [34,29,21,21,34,34,36,32],
              [17,22,15,21,28,25,40,33],
              [39,27,23,40,45,46,72,41],
              [41,26,25,34,47,48,65,38],
              [33,28,22,26,37,34,41,33],
              [14,12, 9,22,27,17,51,23]]

    pack_and_test(a, b, ab)
