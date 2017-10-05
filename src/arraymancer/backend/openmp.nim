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

when defined(openmp):
  {.passC: "-fopenmp".}
  {.passL: "-fopenmp".}

  {. pragma: omp, header:"omp.h" .}

  proc omp_set_num_threads*(x: cint) {.omp.}
  proc omp_get_num_threads*(): cint {.omp.}
  proc omp_get_max_threads*(): cint {.omp.}
  proc omp_get_thread_num*(): cint {.omp.}

else:
  template omp_set_num_threads*(x: cint) = discard
  template omp_get_num_threads*(): cint = 1
  template omp_get_max_threads*(): cint = 1
  template omp_get_thread_num*(): cint = 0

const OMP_FOR_ANNOTATION = "if(ompsize > " & $OMP_FOR_THRESHOLD & ")"

template omp_parallel_countup*(i: untyped, size: Natural, body: untyped): untyped =
  let ompsize = size
  for i in `||`(0, ompsize, OMP_FOR_ANNOTATION):
    body

template omp_parallel_forup*(i: untyped, start, size: Natural, body: untyped): untyped =
  let ompsize = size
  for i in `||`(start, ompsize, OMP_FOR_ANNOTATION):
    body

template omp_parallel_blocks*(block_offset, block_size: untyped, size: Natural, body: untyped): untyped =
  block ompblocks:
    when defined(openmp):
      if size >= OMP_FOR_THRESHOLD:
        let omp_num_threads = omp_get_max_threads()
        if size >= omp_num_threads:
          let bsize = size div omp_num_threads
          for j in 0||(omp_num_threads-1):
            let block_offset = bsize*j
            let block_size = if j < omp_num_threads-1: bsize else: size - block_offset
            block:
              body
          break ompblocks
    let block_offset = 0
    let block_size = size
    block:
      body
