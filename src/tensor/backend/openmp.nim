# Copyright 2017 the Arraymancer contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import  ./global_config,
        ./memory_optimization_hints

when defined(openmp):
  when not defined(cuda): # For cuda, OpenMP flags must be passeed
    {.passC: "-fopenmp".} # behind -Xcompiler -fopenmp
    {.passL: "-fopenmp".}

  {.pragma: omp, header:"omp.h".}

  proc omp_set_num_threads*(x: cint) {.omp.}
  proc omp_get_num_threads*(): cint {.omp.}
  proc omp_get_max_threads*(): cint {.omp.}
  proc omp_get_thread_num*(): cint {.omp.}

else:
  template omp_set_num_threads*(x: cint) = discard
  template omp_get_num_threads*(): cint = 1
  template omp_get_max_threads*(): cint = 1
  template omp_get_thread_num*(): cint = 0

const OMP_FOR_ANNOTATION = "simd if(ompsize > " & $OMP_FOR_THRESHOLD & ")"

template omp_parallel_countup*(i: untyped, size: Natural, body: untyped): untyped =
  let ompsize = size
  for i in `||`(0, ompsize, OMP_FOR_ANNOTATION):
    body

template omp_parallel_forup*(i: untyped, start, size: Natural, body: untyped): untyped =
  let ompsize = size
  for i in `||`(start, ompsize, OMP_FOR_ANNOTATION):
    body

template omp_parallel_blocks*(block_offset, block_size: untyped, size: Natural, body: untyped): untyped =
  if likely(size > 0):
    block ompblocks:
      when defined(openmp):
        if size >= OMP_FOR_THRESHOLD:
          let num_blocks = min(omp_get_max_threads(), size)
          if num_blocks > 1:
            let bsize = size div num_blocks
            for block_index in `||`(0, num_blocks-1, "simd"):
              # block_offset and block_size are injected into the calling proc
              let block_offset = bsize*block_index
              let block_size = if block_index < num_blocks-1: bsize else: size - block_offset
              block:
                body
            break ompblocks

      # block_offset and block_size are injected into the calling proc
      let block_offset = 0
      let block_size = size
      block:
        body


template omp_parallel_reduce_blocks*[T](reduced: T, block_offset, block_size: untyped, size, weight: Natural, op_final, op_init, op_middle: untyped): untyped =


  # To prevent false sharing, results will be stored in an array but
  # padded to be a cache line apart atleast.
  # All CPUs cache line is 64B, 16 float32/int32 fits or 8 float64/int64

  # TODO compile time evaluation depending of sizeof(T)
  # Pending https://github.com/nim-lang/Nim/pull/5664
  const maxItemsPerCacheLine = 16

  if likely(size > 0):
    block ompblocks:
      when defined(openmp):
        if size*weight >= OMP_FOR_THRESHOLD:
          let num_blocks = min(min(size, omp_get_max_threads()), OMP_MAX_REDUCE_BLOCKS)
          if num_blocks > 1:
            withMemoryOptimHints()
            var results{.align64.}: array[OMP_MAX_REDUCE_BLOCKS * maxItemsPerCacheLine, type(reduced)]
            let bsize = size div num_blocks

            if bsize > 1:
              # Initialize first elements
              for block_index in 0..<num_blocks:
                # block_offset and block_size are injected into the calling proc
                let block_offset = bsize*block_index
                let block_size = if block_index < num_blocks-1: bsize else: size - block_offset

                # Inject x using a template to able to mutate it
                template x(): untyped =
                  results[block_index * maxItemsPerCacheLine]

                block:
                  op_init

              # Reduce blocks
              for block_index in `||`(0, num_blocks-1, "simd"):
                # block_offset and block_size are injected into the calling proc
                var block_offset = bsize*block_index
                let block_size = (if block_index < num_blocks-1: bsize else: size - block_offset) - 1
                block_offset += 1

                # Inject x using a template to able to mutate it
                template x(): untyped =
                  results[block_index * maxItemsPerCacheLine]

                block:
                  op_middle

              # Finally reduce results from openmp
              block:
                shallowCopy(reduced, results[0])

                # Inject x using a template to able to mutate it
                template x(): untyped =
                  reduced

                for block_index in 1..<num_blocks:
                  let y {.inject.} = results[block_index * maxItemsPerCacheLine]
                  op_final

              break ompblocks

      # Fallback normal sequential reduce
      block:
        # Initialize first elements
        var block_offset = 0
        block:
          template x(): untyped =
            reduced

          block:
            op_init

        # Offset to reduce rest of elements
        block_offset = 1
        let block_size = size-1

        if block_size > 0:
          # Inject x using a template to able to mutate it
          template x(): untyped =
            reduced
          block:
            op_middle