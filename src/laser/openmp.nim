# Laser
# Copyright (c) 2018 Mamy André-Ratsimbazafy
# Distributed under the Apache v2 License (license terms are at http://www.apache.org/licenses/LICENSE-2.0).
# This file may not be copied, modified, or distributed except according to those terms.

# ###############################################################
# Compile-time name mangling for OpenMP thresholds
# Workaround https://github.com/nim-lang/Nim/issues/9365
# and https://github.com/nim-lang/Nim/issues/9366
import random
from strutils import toHex

var mangling_rng {.compileTime.} = initRand(0x1337DEADBEEF)
var current_suffix {.compileTime.} = ""

proc omp_suffix*(genNew: static bool = false): string {.compileTime.} =
  ## genNew:
  ##   if false, return the last suffix
  ##   else return a fresh one
  # This is exported because you cannot bind the symbol early enough
  # for exportc

  if genNew:
    current_suffix = mangling_rng.rand(high(uint32)).toHex
  result = current_suffix

# ################################################################
# Tuning

when defined(openmp):
  when not defined(cuda):
    {.passC: "-fopenmp".}
    {.passL: "-fopenmp".}

  {.pragma: omp, header:"omp.h".}

  proc omp_set_num_threads*(x: cint) {.omp.}
  proc omp_get_num_threads*(): cint {.omp.}
  proc omp_get_max_threads*(): cint {.omp.} # This takes hyperthreading into account
  proc omp_get_thread_num*(): cint {.omp.}
  proc omp_set_nested*(x: cint) {.omp.}
  proc omp_get_nested*(): cint {.omp.}

else:
  template omp_set_num_threads*(x: cint) = discard
  template omp_get_num_threads*(): cint = 1
  template omp_get_max_threads*(): cint = 1
  template omp_get_thread_num*(): cint = 0
  template omp_set_nested*(x: cint) = discard
  template omp_get_nested*(): cint = cint 0

# TODO tuning for architectures
# https://github.com/zy97140/omp-benchmark-for-pytorch
# https://github.com/zy97140/omp-benchmark-for-pytorch/blob/master/benchmark-data/IntelR-XeonR-CPU-E5-2669-v4.md
# https://github.com/zy97140/omp-benchmark-for-pytorch/blob/master/benchmark-data/IntelR-XeonR-Platinum-8180-CPU.md


const OMP_MEMORY_BOUND_GRAIN_SIZE*{.intdefine.} = 1024
  ## This is the minimum amount of work per physical cores
  ## for memory-bound processing.
  ## - "copy" and "addition" are considered memory-bound
  ## - "float division" can be considered 2x~4x more complex
  ##   and should be scaled down accordingly
  ## - "exp" and "sin" operations are compute-bound and
  ##   there is a perf boost even when processing
  ##   only 1000 items on 28 cores
  ##
  ## Launching 2 threads per core (HyperThreading) is probably desirable:
  ##   - https://medium.com/data-design/destroying-the-myth-of-number-of-threads-number-of-physical-cores-762ad3919880
  ##
  ## Raising the following parameters can have the following impact:
  ##   - number of sockets: higher, more over memory fetch
  ##   - number of memory channel: lower, less overhead per memory fetch
  ##   - RAM speed: lower, less overhead per memory fetch
  ##   - Private L2 cache: higher, feed more data per CPU
  ##   - Hyperthreading and cache associativity
  ##   - Cores, shared L3 cache: Memory contention
  ##
  ## Note that setting num_threads manually might impact performance negatively:
  ##   - http://studio.myrian.fr/openmp-et-num_threads/
  ##     > 2x2ms overhead when changing num_threads from 16->6->16

const OMP_NON_CONTIGUOUS_SCALE_FACTOR*{.intdefine.} = 4
  ## Due to striding computation, we can use a lower grainsize
  ## for non-contiguous tensors

# ################################################################

template attachGC*(): untyped =
  ## If you are allocating reference types, sequences or strings
  ## in a parallel section, you need to attach and detach
  ## a GC for each thread. Those should be thread-local temporaries.
  ##
  ## This attaches the GC.
  ##
  ## Note: this creates too strange error messages
  ## when --threads is not on: https://github.com/nim-lang/Nim/issues/9489
  if(omp_get_thread_num()!=0):
    setupForeignThreadGc()

template detachGC*(): untyped =
  ## If you are allocating reference types, sequences or strings
  ## in a parallel section, you need to attach and detach
  ## a GC for each thread. Those should be thread-local temporaries.
  ##
  ## This detaches the GC.
  ##
  ## Note: this creates too strange error messages
  ## when --threads is not on: https://github.com/nim-lang/Nim/issues/9489
  if(omp_get_thread_num()!=0):
    teardownForeignThreadGc()

template omp_parallel*(body: untyped): untyped =
  ## Starts an openMP parallel section
  ##
  ## Don't forget to use attachGC and detachGC if you are allocating
  ## sequences, strings, or reference types.
  ## Those should be thread-local temporaries.
  # New line intentional: https://github.com/mratsim/Arraymancer/issues/407
  {.emit: ["""
  #pragma omp parallel """].}
  block: body

template omp_parallel_if*(condition: bool, body: untyped) =
  let predicate = condition # Make symbol valid and ensure it's lvalue
  # New line intentional: https://github.com/mratsim/Arraymancer/issues/407
  {.emit: ["""
  #pragma omp parallel if (", predicate, ")"""].}
  block: body

template omp_for*(
    index: untyped,
    length: Natural,
    use_simd, nowait: static bool,
    body: untyped
  ) =
  ## OpenMP for loop (not parallel)
  ##
  ## This must be used in an `omp_parallel` block
  ## for parallelization.
  ##
  ## Inputs:
  ##   - `index`, the iteration index, similar to
  ##     for `index` in 0 ..< length:
  ##       doSomething(`index`)
  ##   - `length`, the number of elements to iterate on
  ##   - `use_simd`, instruct the compiler to unroll the loops for `simd` use.
  ##     For example, for float32:
  ##     for i in 0..<16:
  ##       x[i] += y[i]
  ##     will be unrolled to take 128, 256 or 512-bit to use SSE, AVX or AVX512.
  ##     for 256-bit AVX:
  ##     for i in countup(0, 2, 8): # Step 8 by 8
  ##       x[i]   += y[i]
  ##       x[i+1] += y[i+1]
  ##       x[i+2] += y[i+2]
  ##       ...
  const omp_annotation = block:
    "for " &
      (when use_simd: "simd " else: "") &
      (when nowait: "nowait " else: "")
  for `index`{.inject.} in `||`(0, length-1, omp_annotation):
    block: body

template omp_parallel_for*(
    index: untyped,
    length: Natural,
    omp_grain_size: static Natural,
    use_simd: static bool,
    body: untyped
  ) =
  ## Parallel for loop
  ##
  ## Do not forget to use attachGC and detachGC if you are allocating
  ## sequences, strings, or reference types.
  ## Those should be thread-local temporaries.
  ##
  ## Inputs:
  ##   - `index`, the iteration index, similar to
  ##     for `index` in 0 ..< length:
  ##       doSomething(`index`)
  ##   - `length`, the number of elements to iterate on
  ##   - `omp_grain_size`, the minimal amount of work per thread. If below,
  ##     we don't start threads. Note that we always start as much hardware threads
  ##     as available as starting varying number of threads in the lifetime of the program
  ##     will add oberhead.
  ##   - `use_simd`, instruct the compiler to unroll the loops for `simd` use.
  ##     For example, for float32:
  ##     for i in 0..<16:
  ##       x[i] += y[i]
  ##     will be unrolled to take 128, 256 or 512-bit to use SSE, AVX or AVX512.
  ##     for 256-bit AVX:
  ##     for i in countup(0, 2, 8): # Step 8 by 8
  ##       x[i]   += y[i]
  ##       x[i+1] += y[i+1]
  ##       x[i+2] += y[i+2]
  ##       ...
  when not defined(openmp):
    ## When OpenMP is not defined we use this simple loop as fallback
    ## This way, the compiler will still be provided "simd" vectorization hints
    when use_simd:
      const omp_annotation = "parallel for simd"
    else:
      const omp_annotation = "parallel for"
    for `index`{.inject.} in `||`(0, length-1, omp_annotation):
      block: body
  else:
    let omp_size = length # make sure if length is computed it's only done once

    const # Workaround to expose an unique symbol in C. TODO Pending OpenMP interpolation: https://github.com/nim-lang/Nim/issues/9365
      omp_condition_csym = "omp_condition_" & omp_suffix(genNew = true)
    let omp_condition {.exportc: "omp_condition_" & # We cannot use csym directly in exportc
      omp_suffix().} = omp_grain_size * omp_get_max_threads() < omp_size

    const omp_annotation = block:
      "parallel for " &
        (when use_simd: "simd " else: "") &
        "if(" & $omp_condition_csym & ")"

    for `index`{.inject.} in `||`(0, omp_size - 1, omp_annotation):
      block: body

template omp_parallel_for_default*(
      index: untyped,
      length: Natural,
      body: untyped
      ) =
  ## This will be renamed omp_parallel_for once
  ## https://github.com/nim-lang/Nim/issues/9414 is solved.
  ## Compared to omp_parallel_for the following are set by default
  ## - omp_grain_size:
  ##     The default `OMP_MEMORY_BOUND_GRAIN_SIZE` is suitable for
  ##     contiguous copy or add operations. It's 1024 and can be changed
  ##     by passing `-d:OMP_MEMORY_BOUND_GRAIN_SIZE=123456` during compilation.
  ##     A value of 1 will always parallelize the loop.
  ## - simd is used by default
  omp_parallel_for(
        index,
        length,
        omp_grain_size = OMP_MEMORY_BOUND_GRAIN_SIZE,
        use_simd = true,
        body
        )

template omp_chunks*(
    omp_size: Natural, #{lvalue} # TODO parameter constraint, pending https://github.com/nim-lang/Nim/issues/9620
    chunk_offset, chunk_size: untyped,
    body: untyped): untyped =
  ## Internal proc
  ## This is is the chunk part of omp_parallel_chunk
  ## omp_size should be a lvalue (assigned value) and not
  ## the result of a routine otherwise routine and its side-effect will be called multiple times

  # The following simple chunking scheme can lead to severe load imbalance
  #
  # `chunk_offset`{.inject.} = chunk_size * thread_id
  # `chunk_size`{.inject.} =  if thread_id < nb_chunks - 1: chunk_size
  #                           else: omp_size - chunk_offset
  #
  # For example dividing 40 items on 12 threads will lead to
  # a base_chunk_size of 40/12 = 3 so work on the first 11 threads
  # will be 3 * 11 = 33, and the remainder 7 on the last thread.
  let
    nb_chunks = omp_get_num_threads()
    base_chunk_size = omp_size div nb_chunks
    remainder = omp_size mod nb_chunks
    thread_id = omp_get_thread_num()

  # Instead of dividing 40 work items on 12 cores into:
  # 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 7 = 3*11 + 7 = 40
  # the following scheme will divide into
  # 4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3 = 4*4 + 3*8 = 40
  #
  # This is compliant with OpenMP spec (page 60)
  # http://www.openmp.org/mp-documents/openmp-4.5.pdf
  # "When no chunk_size is specified, the iteration space is divided into chunks
  # that are approximately equal in size, and at most one chunk is distributed to
  # each thread. The size of the chunks is unspecified in this case."
  # ---> chunks are the same ±1

  var `chunk_offset`{.inject.}, `chunk_size`{.inject.}: Natural
  if thread_id < remainder:
    chunk_offset = (base_chunk_size + 1) * thread_id
    chunk_size = base_chunk_size + 1
  else:
    chunk_offset = base_chunk_size * thread_id + remainder
    chunk_size = base_chunk_size

  block: body

template omp_parallel_chunks*(
    length: Natural,
    chunk_offset, chunk_size: untyped,
    omp_grain_size: static Natural,
    body: untyped): untyped =
  ## Create a chunk for each thread. You can use:
  ## `for index in chunk_offset ..< chunk_size:` or
  ## `zeroMem(foo[chunk_offset].addr, chunk_size)`
  ##
  ## Splits the input `length` into chunks and do a parallel loop
  ## on each chunk. The number of chunks depends on the number of cores at runtime.
  ## `chunk_offset` and `chunk_size` should be passed as undeclared identifiers.
  ## Within the template scope they will contain the start offset and the length
  ## of the current thread chunk. I.e. their value is thread-specific.
  ##
  ## Use omp_get_thread_num() to get the current thread number
  ##
  ## This is useful for non-contiguous processing as a replacement to omp_parallel_for
  ## or when operating on (contiguous) ranges for example for memset or memcpy.
  ##
  ## Do not forget to use attachGC and detachGC if you are allocating
  ## sequences, strings, or reference types.
  ## Those should be thread-local temporaries.
  when not defined(openmp):
    const `chunk_offset`{.inject.} = 0
    let `chunk_size`{.inject.} = length
    block: body
  else:
    let omp_size = length # make sure if length is computed it's only done once
    let over_threshold = omp_grain_size * omp_get_max_threads() < omp_size

    omp_parallel_if(over_threshold):
      omp_chunks(omp_size, chunk_offset, chunk_size, body)

template omp_parallel_chunks_default*(
    length: Natural,
    chunk_offset, chunk_size: untyped,
    body: untyped): untyped =
  ## This will be renamed omp_parallel_chunks once
  ## https://github.com/nim-lang/Nim/issues/9414 is solved.
  ## Compared to omp_parallel_for the following are set by default
  ## - omp_grain_size:
  ##     The default `OMP_MEMORY_BOUND_GRAIN_SIZE` is suitable for
  ##     contiguous copy or add operations. It's 1024 and can be changed
  ##     by passing `-d:OMP_MEMORY_BOUND_GRAIN_SIZE=123456` during compilation.
  ##     A value of 1 will always parallelize the loop.
  omp_parallel_chunks(
    length,
    chunk_offset, chunk_size,
    omp_grain_size = OMP_MEMORY_BOUND_GRAIN_SIZE,
    body
  )

# New line intentional: https://github.com/mratsim/Arraymancer/issues/407

template omp_critical*(body: untyped): untyped =
  {.emit: ["""
  #pragma omp critical"""].}
  block: body

template omp_master*(body: untyped): untyped =
  {.emit: ["""
  #pragma omp master"""].}
  block: body

template omp_single*(body: untyped): untyped =
  {.emit: ["""
  #pragma omp single"""].}
  block: body

template omp_single_nowait*(body: untyped): untyped =
  {.emit: ["""
  #pragma omp single nowait"""].}
  block: body

template omp_barrier*(): untyped =
  {.emit: ["""
  #pragma omp barrier"""].}

template omp_task*(annotation: static string, body: untyped): untyped =
  {.emit: [
    """#pragma omp task """, annotation].}
  block: body

template omp_taskwait*(): untyped =
  {.emit: ["""
  #pragma omp taskwait"""].}

template omp_taskloop*(
    index: untyped,
    length: Natural,
    annotation: static string,
    body: untyped
  ) =
  ## OpenMP taskloop
  const omp_annotation = "taskloop " & annotation
  for `index`{.inject.} in `||`(0, length-1, omp_annotation):
    block: body

import macros
macro omp_flush*(variables: varargs[untyped]): untyped =
  var listvars = "("
  for i, variable in variables:
    if i == 0:
      listvars.add "`" & $variable & "`"
    else:
      listvars.add ",`" & $variable & "`"
  listvars.add ')'
  result = quote do:
    {.emit: ["""
    #pragma omp flush " & `listvars`"""].}
