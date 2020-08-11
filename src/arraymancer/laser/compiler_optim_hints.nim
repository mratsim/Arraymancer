# Laser & Arraymancer
# Copyright (c) 2017-2018 Mamy André-Ratsimbazafy
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

const LASER_MEM_ALIGN*{.intdefine.} = 64
static:
  assert LASER_MEM_ALIGN != 0, "Alignment " & $LASER_MEM_ALIGN & "must be a power of 2"
  assert (LASER_MEM_ALIGN and (LASER_MEM_ALIGN - 1)) == 0, "Alignment " & $LASER_MEM_ALIGN & "must be a power of 2"

template withCompilerOptimHints*() =
  # See https://gcc.gnu.org/onlinedocs/gcc/Common-Function-Attributes.html
  # and https://gcc.gnu.org/onlinedocs/gcc/Common-Variable-Attributes.html#Common-Variable-Attributes

  # Variable is created aligned by LASER_MEM_ALIGN.
  # This is useful to ensure an object can be loaded
  # in a minimum amount of cache lines load
  # For example, the stack part of tensors are 128 bytes and can be loaded in 2 cache lines
  # but would require 3 loads if they are misaligned.
  {.pragma: align_variable, codegenDecl: "$# $# __attribute__((aligned(" & $LASER_MEM_ALIGN & ")))".}

  # Variable. Pointer does not alias any existing valid pointers.
  when not defined(vcc):
    {.pragma: restrict, codegenDecl: "$# __restrict__ $#".}
  else:
    {.pragma: restrict, codegenDecl: "$# __restrict $#".}

const withBuiltins = defined(gcc) or defined(clang) or defined(icc)

type
  PrefetchRW* {.size: cint.sizeof.} = enum
    Read = 0
    Write = 1
  PrefetchLocality* {.size: cint.sizeof.} = enum
    NoTemporalLocality = 0 # Data can be discarded from CPU cache after access
    LowTemporalLocality = 1
    ModerateTemporalLocality = 2
    HighTemporalLocality = 3 # Data should be left in all levels of cache possible
    # Translation
    # 0 - use no cache eviction level
    # 1 - L1 cache eviction level
    # 2 - L2 cache eviction level
    # 3 - L1 and L2 cache eviction level

when withBuiltins:
  proc builtin_assume_aligned(data: pointer, alignment: csize): pointer {.importc: "__builtin_assume_aligned", noDecl.}
  proc builtin_prefetch(data: pointer, rw: PrefetchRW, locality: PrefetchLocality) {.importc: "__builtin_prefetch", noDecl.}

when defined(cpp):
  proc static_cast[T: ptr](input: pointer): T
    {.importcpp: "static_cast<'0>(@)".}

template assume_aligned*[T](data: ptr T, alignment: static int = LASER_MEM_ALIGN): ptr T =
  when defined(cpp) and withBuiltins: # builtin_assume_aligned returns void pointers, this does not compile in C++, they must all be typed
    static_cast[ptr T](builtin_assume_aligned(data, alignment))
  elif withBuiltins:
    cast[ptr T](builtin_assume_aligned(data, alignment))
  else:
    data

template prefetch*[T](
            data: ptr (T or UncheckedArray[T]),
            rw: static PrefetchRW = Read,
            locality: static PrefetchLocality = HighTemporalLocality) =
  ## Prefetch examples:
  ##   - https://scripts.mit.edu/~birge/blog/accelerating-code-using-gccs-prefetch-extension/
  ##   - https://stackoverflow.com/questions/7327994/prefetching-examples
  ##   - https://lemire.me/blog/2018/04/30/is-software-prefetching-__builtin_prefetch-useful-for-performance/
  ##   - https://www.naftaliharris.com/blog/2x-speedup-with-one-line-of-code/
  when withBuiltins:
    builtin_prefetch(data, rw, locality)
  else:
    discard

template pragma_ivdep() {.used.} =
  ## Tell the compiler to ignore unproven loop dependencies
  ## such as "a[i] = a[i + k] * c;" if k is unknown, as it introduces a loop
  ## dependency if it's negative
  ## https://software.intel.com/en-us/node/524501
  ##
  ## Placeholder
  # We don't expose that as it only works on C for loop. Nim only generates while loop
  # except when using OpenMP. But the OpenMP "simd" already achieves the same as ivdep.
  when defined(gcc):
    {.emit: "#pragma GCC ivdep".}
  else: # Supported on ICC and Cray
    {.emit: "pragma ivdep".}

template withCompilerFunctionHints() {.used.} =
  ## Not exposed, Nim codegen will declare them as normal C function.
  ## This messes up with N_NIMCALL, N_LIB_PRIVATE, N_INLINE and also
  ## creates duplicate symbols when one function called by a hot or pure function
  ## is public and inline (because hot and pure cascade to all cunfctions called)
  ## and they cannot be stacked easily: (hot, pure) will only apply the last

  # Function. Returned pointer is aligned to LASER_MEM_ALIGN
  {.pragma: aligned_ptr_result, codegenDecl: "__attribute__((assume_aligned(" & $LASER_MEM_ALIGN & ")) $# $#$#".}

  # Function. Returned pointer cannot alias any other valid pointer and no pointers to valid object occur in any
  # storage pointed to.
  {.pragma: malloc, codegenDecl: "__attribute__((malloc)) $# $#$#".}

  # Function. Creates one or more function versions that can process multiple arguments using SIMD.
  # Ignored when -fopenmp is used and within an OpenMP simd loop
  {.pragma: simd, codegenDecl: "__attribute__((simd)) $# $#$#".}

  # Function. Indicates hot and cold path. Ignored when using profile guided optimization.
  {.pragma: hot, codegenDecl: "__attribute__((hot)) $# $#$#".}
  {.pragma: cold, codegenDecl: "__attribute__((cold)) $# $#$#".}

  # ## pure and const
  # ## Affect Common Sub-expression Elimination, Dead Code Elimination and loop optimization.
  # See
  #   - https://lwn.net/Articles/285332/
  #   - http://benyossef.com/helping-the-compiler-help-you/
  #
  # Function. The function only accesses its input params and global variables state.
  # It does not modify any global, calling it multiple times with the same params
  # and global variables will produce the same result.
  {.pragma: gcc_pure, codegenDecl: "__attribute__((pure)) $# $#$#".}
  #
  # Function. The function only accesses its input params and calling it multiple times
  # with the same params will produce the same result.
  # Warning ⚠:
  #   Pointer inputs must not be dereferenced to read the memory pointed to.
  #   In Nim stack arrays are passed by pointers and big stack data structures
  #   are passed by reference as well. I.e. Result unknown.
  {.pragma: gcc_const, codegenDecl: "__attribute__((const)) $# $#$#".}

  # We don't define per-function fast-math, GCC attribute optimize is broken:
  # --> https://gcc.gnu.org/ml/gcc/2009-10/msg00402.html
  #
  # Workaround floating point latency for algorithms like sum
  # should be done manually.
  #
  # See : https://stackoverflow.com/questions/39095993/does-each-floating-point-operation-take-the-same-time
  # and https://www.agner.org/optimize/vectorclass.pdf "Using multiple accumulators"
  #
  # FP addition has a latency of 3~5 clock cycles, i.e. the result cannot be reused for that much time.
  # But the throughput is 1 FP add per clock cycle (and even 2 per clock cycle for Skylake)
  # So we need to use extra accumulators to fully utilize the FP throughput despite FP latency.
  # On Skylake, all FP latencies are 4: https://www.agner.org/optimize/blog/read.php?i=415
  #
  # Note that this is per CPU cores, each core needs its own "global CPU accumulator" to combat
  # false sharing when multithreading.
  #
  # This wouldn't be needed with fast-math because compiler would consider FP addition associative
  # and create intermediate variables as needed to exploit this through put.
