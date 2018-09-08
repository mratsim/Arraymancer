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

import  ../tensor/backend/memory_optimization_hints,
        ../tensor/backend/openmp,
        ../tensor/tensor,
        ./private/p_nnp_types,
        ./fallback/conv

proc maxpool2d*[T](input: Tensor[T],
                kernel: Size2D,
                padding: Size2D = (0,0),
                stride: Size2D = (1,1)
                ): tuple[max_indices: Tensor[int], maxpooled: Tensor[T]] {.noinit.}=
  ## MaxPool 2D forward pass

  assert input.rank == 4 and input.is_C_contiguous

  let
    N = input.shape[0]
    C = input.shape[1]
    H = input.shape[2]
    W = input.shape[3]

    kH = kernel.height
    kW = kernel.width

    outH = (H + (2 * padding.height) - kH) div stride.height + 1
    outW = (W + (2 * padding.width ) - kW) div stride.width  + 1

  result.max_indices = newTensoruninit[int](N * C * outH * outW)
  result.maxpooled   = newTensoruninit[ T ](N, C, outH, outW)

  withMemoryOptimHints()
  let idata {.restrict.} = input.dataArray
  let idx_data {.restrict.} = result.max_indices.dataArray # Warning ⚠: data pointed to will be mutated
  let max_data {.restrict.} = result.maxpooled.dataArray # Warning ⚠: data pointed to will be mutated

  for n in `||`(0, N-1, "simd"):
    for c in 0 ..< C:
      for h in 0 ..< outH:
        for w in 0 ..< outW:
          var max = low(T)
          var argmax = low(int)
          for ph in 0 ..< kH:
            let row = h * stride.height + ph - padding.height
            if 0 <= row and row < H:
              for pw in 0 ..< kW:
                let col = w * stride.width + pw - padding.width
                if 0 <= col and col < W:
                  let iidx = col + W * (row + H * (c + n * C))
                  let val = idata[iidx]
                  if val > max:
                    max = val
                    argmax = iidx
          let oidx = w + outW * (h + outH * (c + n * C))
          max_data[oidx] = max
          idx_data[oidx] = argmax

proc maxpool2d_backward*[T](
  cached_input_shape: openarray[int]|MetadataArray,
  cached_max_indices: Tensor[int],
  gradOutput: Tensor[T]
  ): Tensor[T] {.noInit.}=

  assert gradOutput.size == cached_max_indices.size

  result = zeros[T](cached_input_shape) # gradInput

  let rdata = result.dataArray # Warning ⚠: data pointed to will be mutated
  let godata = gradOutput.dataArray
  let cmidata = cached_max_indices.dataArray

  omp_parallel_countup(i, gradOutput.size - 1):
    rdata[cmidata[i]] = godata[i]
