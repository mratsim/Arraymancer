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


import  ../backend/openmp,
        ../backend/memory_optimization_hints,
        ../data_structure,
        ../init_cpu,
        ../higher_order_applymap

proc cmp_idx_max*[T](accum: var Tensor[tuple[idx: int, max: T]],
                    next_idx: int,
                    next: Tensor[T]) =
  ## Compare a tensor containing accumulated (idx_of_maxval, max_value)
  ## and another tensor at a specified index
  ## Store the max value and its corresponding index in the accumulator
  ##
  ##
  ## Necessary for argmax, core computation step
  apply2_inline(accum, next):
    if x.max < y:
      (next_idx, y)
    else:
      x

proc cmp_idx_max*[T](accum: var Tensor[tuple[idx: int, max: T]],
                    next: Tensor[tuple[idx: int, max: T]]) =
  ## Compare two tensors containing accumulated (idx_of_maxval, max_value)
  ## Store the max value and its corresponding index in the first accumulator
  ##
  ## Necessary for argmax, merge partial folds step
  apply2_inline(accum, next):
    if x.max < y.max:
      y
    else:
      x

template unzip_idx_max*[T](t: Tensor[tuple[idx: int, max: T]], op:untyped): untyped =
  # Type signature pending https://github.com/nim-lang/Nim/issues/4061

  var dest = (indices: newTensorUninit[int](t.shape),
              maxes:   newTensorUninit[T](t.shape))
  withMemoryOptimHints()
  var
    data1{.restrict.} = dest.indices.dataArray
    data2{.restrict.} = dest.maxes.dataArray

  omp_parallel_blocks(block_offset, block_size, dest.indices.size):
    for i, x {.inject.} in enumerate(t, block_offset, block_size):
      (data1[i], data2[i]) = op
  dest