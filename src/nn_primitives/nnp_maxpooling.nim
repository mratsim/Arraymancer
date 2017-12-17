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

import  ../tensor/tensor,
        ./private/p_nnp_types,
        ./fallback/conv

proc maxpool2d*[T](input: Tensor[T],
                kernel: Size2D,
                padding: Size2D = (0,0),
                stride: Size2D = (1,1),
                argmax: var Tensor[int],
                result: var Tensor[T]
                ) =
  ## MaxPool 2D forward pass

  assert input.rank == 4

  let
    N = input.shape[0]
    C = input.shape[1]
    H = input.shape[2]
    W = input.shape[3]

    kH = kernel.height
    kW = kernel.width

    outH = (H + (2 * padding.height) - kH) div stride.height + 1
    outW = (W + (2 * padding.width ) - kW) div stride.width  + 1

    channels_col = C * kH * kW
    flatten_size_col = outH * outW

  var x_cols = newTensorUninit[T](channels_col, flatten_size_col)
  let x_split = input.reshape(N * C, 1, H, W)

  im2col(x_split, (kH, kW), padding, stride, -Inf.T, x_cols) # TODO: replace by low(T) when 0.18 for https://github.com/nim-lang/Nim/commit/badba83d38371726bafba5870d5fb927eb453e41

  (argmax, result) = x_cols.argmax(axis = 0)
  result = result.reshape(outH, outW, N, C).permute(2, 3, 0, 1)