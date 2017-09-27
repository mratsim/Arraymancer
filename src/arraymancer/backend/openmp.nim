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

const OMP_FOR_ANNOTATION = "if(ompsize > " & $OMP_FOR_THRESHOLD & ")"

template omp_parallel_countup*(i: untyped, size: Natural, body: untyped): untyped =
  let ompsize = size
  for i in `||`(0, ompsize, OMP_FOR_ANNOTATION):
    body

template omp_parallel_forup*(i: untyped, start, size: Natural, body: untyped): untyped =
  let ompsize = size
  for i in `||`(start, ompsize, OMP_FOR_ANNOTATION):
    body

const num_omp_parallel_blocks = 4 # TODO: use number of OMP threads instead

template omp_parallel_blocks*(block_offset, block_size: untyped, size: Natural, body: untyped): untyped =
  when defined(openmp):
    block:
      if size >= OMP_FOR_THRESHOLD and size >= num_omp_parallel_blocks:
        let bsize = size div num_omp_parallel_blocks
        for j in 0||(num_omp_parallel_blocks-1):
          let block_offset = bsize*j
          let block_size = if j < num_omp_parallel_blocks-1: bsize else: size - block_offset
          block:
            body
        break
  let block_offset = 0
  let block_size = size
  block:
    body
