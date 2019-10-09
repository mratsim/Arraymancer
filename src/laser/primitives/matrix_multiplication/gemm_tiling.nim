# Laser
# Copyright (c) 2018 Mamy André-Ratsimbazafy
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

# ############################################################
#
#               Cache and register optimizations
#
# ############################################################

# Papers:
#   [1] Anatomy of High-Performance Matrix Multiplication (Revised)
#       Kazushige Goto, Robert A. Van de Geijn
#     - http://www.cs.utexas.edu/~flame/pubs/GotoTOMS_revision.pdf
#
#   [2] Anatomy of High-Performance Many-Threaded Matrix Multiplication
#       Smith et al
#     - http://www.cs.utexas.edu/users/flame/pubs/blis3_ipdps14.pdf
#
#   [3] Automating the Last-Mile for High Performance Dense Linear Algebra
#       Veras et al
#     - https://arxiv.org/pdf/1611.08035.pdf
#
#   [4] GEMM: From Pure C to SSE Optimized Micro Kernels
#       Michael Lehn
#     - http://apfel.mathematik.uni-ulm.de/~lehn/sghpc/gemm/index.html
#
#   Laser wiki - GEMM optimization resources
#     - https://github.com/numforge/laser/wiki/GEMM-optimization-resources

import
  # ../../cpuinfo,
  ../../compiler_optim_hints,
  ../../private/[memory, align_unroller],
  typetraits, macros,
  ./gemm_utils

# ############################################################
#
#                    Microkernel (µkernel)
#
# ############################################################

# We have to take into account vectorisation
# so that the microkernel can be processed with vectorized intrinsics.
#
# Caux [mr, nr] must stay in register.
#   - mr ~= nr is optimal to amortize register load cost
#   - some registers must be left to prefetch Ã and ~B (PackedA and PackedB)
#   - nr >= (flops/cycle) / (bytes/cycle) * sizeof(element)
#
# For example Haswell is capable of
#   - 32 single-precision FLOPs/cycle
#   - 32 bytes/cycle store and 64 bytes/cycle load (store C, load A and B)
#
#   so nr >= 32/32 * 4
#   For that number of FLOP it must schedule
#   2xFMA so consume 16 single-precision float
#   so mr*nr >= 16

type
  MicroKernel* = object
    mr*, nr*: int
    cpu_simd*: CPUFeatureX86
    nb_scalars*: int # Ideally MicroKernel should be generic over T
    nb_vecs_nr*: int
    c_unit_stride*: bool # We can use SIMD for the epilogue of C has a unit_stride
    pt*: int # Parallelization threshold

    # TODO: ARM support
    #   - https://github.com/nim-lang/Nim/issues/9679
    #   - https://github.com/nim-lang/Nim/issues/9678

  CPUFeatureX86* = enum
    x86_Generic,
    x86_SSE,
    x86_SSE2,
    x86_SSE4_1,
    x86_AVX,
    x86_AVX_FMA,
    x86_AVX2,
    x86_AVX512
    #   Note that Skylake SP, Xeon Bronze Silver and Gold 5XXX
    #   only have a single AVX512 port and AVX2 can be faster
    #   due to AVX512 downclocking

  X86_FeatureMap = array[CPUFeatureX86, int]

const X86_vecwidth_float: X86_FeatureMap = [
  x86_Generic:         1,
  x86_SSE:     128 div 8,
  x86_SSE2:    128 div 8,
  x86_SSE4_1:  128 div 8,
  x86_AVX:     256 div 8,
  x86_AVX_FMA: 256 div 8,
  x86_AVX2:    256 div 8,
  x86_AVX512:  512 div 8
]

const X86_vecwidth_int: X86_FeatureMap = [
  x86_Generic:         1,
  x86_SSE:             1,
  x86_SSE2:    128 div 8,
  x86_SSE4_1:  128 div 8,
  x86_AVX:     128 div 8,  # Not even addition with integer AVX
  x86_AVX_FMA: 128 div 8,
  x86_AVX2:    256 div 8,
  x86_AVX512:  512 div 8
]

# Registers constraints and micro-kernel tuning
#   - To issue 2xFMAs in parallel we need to use 2x SIMD registers
#   - We want to hold C of size MR * NR completely in SIMD registers as well
#     as each value is reused k times during accumulation C[i, j] += A[i, k] * B[k, j]
#   - We should have enough SIMD registers left to hold
#     the corresponding sections of A and B (at least 4, 2xA and 2xB for FMAs)
#
# On x86-64 X SIMD registers that can issue 2xFMAs per cycle:
#    - NbVecs is 2 minimum
#    - RegsPerVec = 2 * NbVecs => 4 minimum (for A and for B)
#    - NR = NbVecs * NbScalarsPerSIMD
#    - C: MR*NR and uses MR*NbVecs SIMD registers
#    - MR*NbVecs + RegsPerVec <= X
#       -> MR*NbVecs + 2 * NbVecs <= X
#       -> (MR+2) * NbVecs <= X
#
# Some solutions:
#    - AVX with 16 registers:
#          - MR = 6, NbVecs = 2
#            FP32: 8xFP32 per SIMD --> NR = 2x8
#                  ukernel = 6x16
#            FP64, ukernel = 6x8
#          - MR = 2, NbVecs = 4
#            FP32: 8xFP32 per SIMD --> NR = 4x8
#                  ukernel = 2x32
#            FP64, ukernel = 2x16
#    - AVX512 with 32 registers
#          - MR = 6, NbVecs = 4
#            FP32 ukernel = 6x64
#            FP64 ukernel = 6x32
#          - MR = 2, NbVecs = 8
#            FP32 ukernel = 2x128
#            FP64 ukernel = 2x64
#          - MR = 14, NbVecs = 2
#            FP32 ukernel = 14x32
#            FP64 ukernel = 14x16
when defined(amd64): # 64-bit
  # MR configuration - rows of Ã in micro kernel
  # 16 General purpose registers
  const X86_regs: X86_FeatureMap = [
    x86_Generic: 2,
    x86_SSE:     6,
    x86_SSE2:    6,
    x86_SSE4_1:  6,
    x86_AVX:     6,
    x86_AVX_FMA: 6,
    x86_AVX2:    6,
    x86_AVX512:  14
  ]

  # NR configuration - Nb of ~B SIMD vectors
  # We will also keep as many rows of Ã in SIMD registers at the same time
  const NbVecs: X86_FeatureMap = [
      x86_Generic: 1,
      x86_SSE:     2, # 16 XMM registers
      x86_SSE2:    2,
      x86_SSE4_1:  2,
      x86_AVX:     2, # 16 YMM registers
      x86_AVX_FMA: 2,
      x86_AVX2:    2,
      x86_AVX512:  2  # 32 ZMM registers
    ]
else: # 32-bit
  # MR configuration - registers for the rows of Ã
  # 8 General purpose registers
  const X86_regs: X86_FeatureMap = [
    x86_Generic: 2,
    x86_SSE:     2,
    x86_SSE2:    2,
    x86_SSE4_1:  2,
    x86_AVX:     2,
    x86_AVX_FMA: 2,
    x86_AVX2:    2,
    x86_AVX512:  2
  ]

  # NR configuration - Nb of ~B SIMD vectors
  const NbVecs: X86_FeatureMap = [
      x86_Generic: 1,
      x86_SSE:     2, # 8 XMM registers
      x86_SSE2:    2,
      x86_SSE4_1:  2,
      x86_AVX:     2, # 8 YMM registers
      x86_AVX_FMA: 2,
      x86_AVX2:    2,
      x86_AVX512:  2  # 8 ZMM registers
    ]

func x86_ukernel*(cpu: CPUFeatureX86, T: typedesc, c_unit_stride: bool): MicroKernel =
  result.cpu_simd = cpu
  result.c_unit_stride = c_unit_stride
  result.pt = 128
  when T is SomeFloat:
    result.nb_scalars = max(1, X86_vecwidth_float[cpu] div T.sizeof)
  elif T is SomeInteger: # Integers
    result.nb_scalars = max(1, X86_vecwidth_int[cpu] div T.sizeof)
  else:
    {.error: "Unsupported type: " & T.type.name.}

  # The inner microkernel loop does:
  #   AB[m][n] = A[m] * B[n]
  # So n should be the vector size
  # if most matrices are row-Major.
  # This avoids dealing with transpose
  # in the inner loop and untranspose in the epilogue

  result.mr = X86_regs[cpu]                 # 2~6 registers for the rows of Ã
  result.nb_vecs_nr = NbVecs[cpu]           # SIMD vectors of B
  result.nr = result.nb_vecs_nr * result.nb_scalars

#############################################
# Workaround "undeclared identifier mr or nr"
# for some reason the compiler cannot access fields in
# the static MicroKernel.

macro extract_mr*(ukernel: static MicroKernel): untyped =
  result = newLit ukernel.mr
macro extract_nr*(ukernel: static MicroKernel): untyped =
  result = newLit ukernel.nr
macro extract_cpu_simd*(ukernel: static MicroKernel): untyped =
  let simd = ukernel.cpu_simd
  result = quote do: CPUFeatureX86(`simd`)
macro extract_nb_scalars*(ukernel: static MicroKernel): untyped =
  result = newLit ukernel.nb_scalars
macro extract_nb_vecs_nr*(ukernel: static MicroKernel): untyped =
  result = newLit ukernel.nb_vecs_nr
macro extract_c_unit_stride*(ukernel: static MicroKernel): untyped =
  result = newLit ukernel.c_unit_stride
macro extract_pt*(ukernel: static MicroKernel): untyped =
  result = newLit ukernel.pt


# ############################################################
#
#                    Loop tiling
#
# ############################################################

# multithreading info in [2] and https://github.com/flame/blis/blob/master/docs/Multithreading.md

type Tiles*[T] = ref object
  a*: ptr UncheckedArray[T]
  b*: ptr UncheckedArray[T]
  mc*, nc*, kc*: int

  # Multithreaded panels
  ic_num_tasks*: int   # For private L1-L2 and shared L3
  upanelA_size*: int   # Each thread uses a different upanel of A

  # Allocation data
  a_alloc_mem: pointer
  b_alloc_mem: pointer
  # The Tiles data structure takes 64-byte = 1 cache-line


proc deallocTiles[T](tiles: Tiles[T]) =
  if not tiles.a_alloc_mem.isNil:
    deallocShared tiles.a_alloc_mem
  if not tiles.b_alloc_mem.isNil:
    deallocShared tiles.b_alloc_mem

func get_num_tiles*(dim_size, tile_size: int): int {.inline.} =
  ## Get the number of tiles along a dimension depending on the tile size
  (dim_size + tile_size - 1) div tile_size

func partitionMNK*(
      ukernel: static MicroKernel,
      T: typedesc,
      M, N, K: Natural,
    ): tuple[mc, nc, kc: int] =

  result.nc = N # We don't partition over N

  # ## Panel sizes
  # - TLB constraint
  #   TA ̃ + 2(TBj + TCj)≤T
  #   Note: optimizing the similar problem mckc/(2mc+2kc)
  #         under the constraint that mckc ≤ K is the problem
  #         of maximizing the area of a rectangle
  #         while minimizing the perimeter,
  #
  # Goto paper [1] section 6.3: choosing kc
  #   - kc should be as large as possible to amortize the mr*nr updates of Cj
  #   - Elements from Bj [kc, nr] must remain in L1 cache.
  #   - kc * nr should occupy less than half the L1 cache
  #     so that Ã and Caux do not evict element of Bj
  #   - Ã [kc, mc] should occupy
  #     a considerable fraction of the L2 cache
  #   In our experience optimal choice is so that "kc" float64 occupy half a page
  #     -> a page is 4096 bytes = 512 float64 -> half a page = 256

  # Goto paper [1] section 6.3: choosing mc
  #   - mc*kc should fill a considerable part of (1) the memory addressable
  #     by the TLB and (2) the L2 cache
  #     In practice mc is chosen so that A occupies about half the smaller of (1) and (2)


  # TODO: heuristics to compute the size
  result.mc = min( 768 div T.sizeof, M)
  result.kc = min(2048 div T.sizeof, K)

proc newTiles*(
        ukernel: static MicroKernel,
        T: typedesc,
        M, N, K: Natural,
        ): Tiles[T] =
  # BLIS paper [2] section II Figure 2:
  #   - kc * nr in L1 cache µkernel
  #   - mc * kc in L2 cache Ã
  #   - kc * nc in L3 cache ~B (no L3 in Xeon Phi ¯\_(ツ)_/¯)
  new result, deallocTiles[T]
  const
    nr = ukernel.nr
    mr = ukernel.mr

  (result.mc, result.nc, result.kc) = ukernel.partitionMNK(T, M, N, K)

  # Parallel config
  # Ic loop parallel means that each thread will share a panel B and pack a different A
  result.ic_num_tasks = get_num_tiles(M, result.mc)

  # Packing
  # During packing the max size is unroll_stop*kc+kc*LR, LR = MR or NR
  result.upanelA_size = result.kc*round_step_up(result.mc, mr)
  let bufA_size = T.sizeof * result.upanelA_size * result.ic_num_tasks
  let bufB_size = T.sizeof * result.kc*round_step_up(result.nc, nr)

  result.a_alloc_mem = allocShared(bufA_size + 63)
  result.b_alloc_mem = allocShared(bufB_size + 63)
  result.a = assume_aligned align_raw_data(T, result.a_alloc_mem)
  result.b = assume_aligned align_raw_data(T, result.b_alloc_mem)
