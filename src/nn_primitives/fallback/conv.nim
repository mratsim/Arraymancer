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

import  ../../tensor/tensor,
        ../private/p_nnp_types

proc im2col*[T]( input: Tensor[T], kernel_size: Size2D,
                padding: Size2D = (0,0), stride: Size2D = (1,1),
                result: var Tensor[T])  =
  ## Convert blocks of an image into columns, useful for preprocessing
  ## an image before convolutions
  let
    channels = input.nchw_channels
    height = input.nchw_height
    width = input.nchw_width
    channels_col = channels * kernel_size.height * kernel_size.width
    height_col = (height + (2 * padding.height) - kernel_size.height) div stride.height + 1
    width_col = (width + (2 * padding.width) - kernel_size.width) div stride.width + 1
    flatten_size_col = height_col * width_col
    flatten_size = height * width

  assert result.is_C_contiguous and input.is_C_contiguous
  assert result.shape == [channels_col, flatten_size_col]

  let odata = result.unsafe_raw_data()
  let idata = input.unsafe_raw_data()
  for c in `||`(0, channels_col-1, "simd"):
    let
      w_offset = (c mod kernel_size.width) - padding.width
      h_offset = ((c div kernel_size.width) mod kernel_size.height) - padding.height
      c_offset = (c div kernel_size.width) div kernel_size.height
    for h in 0..<height_col:
      let
        row = h_offset + (h * stride.height)
        offset_col = h * width_col
      for w in 0..<width_col:
        let col = w_offset + (w * stride.width)
        var v = 0.T
        if row >= 0 and col >= 0 and row < height and col < width:
          let iidx = (c_offset * flatten_size) + row * width + col
          v = idata[iidx]
        let oidx = (c * flatten_size_col) + offset_col + w
        odata[oidx] = v

proc col2im*[T](input: Tensor[T], channels, height, width: int,
                kernel_size: Size2D,
                padding: Size2D = (0,0), stride: Size2D = (1,1)): Tensor[T] =
  ## Convert blocks of an image from columns back to an image, collapsed
  ## pixels are summed
  let
    channels_col = channels * kernel_size.height * kernel_size.width
    height_col = (height + (2 * padding.height) - kernel_size.height) div stride.height + 1
    width_col = (width + (2 * padding.width) - kernel_size.width) div stride.width + 1
  result = zeros[T](channels, height, width)
  for c in 0..<channels_col:
    let
      w_offset = (c mod kernel_size.width) - padding.width
      h_offset = ((c div kernel_size.width) mod kernel_size.height) - padding.height
      c_offset = (c div kernel_size.width) div kernel_size.height
    for h in 0..<height_col:
      let
        row = h_offset + (h * stride.height)
        offset_col = h * width_col
      for w in 0..<width_col:
        let col = w_offset + (w * stride.width)
        if row < 0 or col < 0 or row >= height or col >= width:
          continue
        result[c_offset, row, col] += input[c, offset_col + w]

proc im2colgemm_conv2d*[T](input, kernel, bias: Tensor[T],
                padding: Size2D = (0,0),
                stride: Size2D = (1,1)): Tensor[T] =
  ## Compute cross-correlate for image with the given kernel weights
  # Implementation with ideas from http://cs231n.github.io/convolutional-networks/#conv
  let
    batch_size = input.shape[^4]
    output_channels = kernel.shape[^4]
    kernel_size = (height: kernel.nchw_height, width: kernel.nchw_width)
    output_height = (input.nchw_height + (2*padding.height) - kernel.nchw_height) div stride.height + 1
    output_width = (input.nchw_width + (2*padding.width) - kernel.nchw_width) div stride.width + 1
    channels_col = input.nchw_channels * kernel.nchw_height * kernel.nchw_width
    kernel_col = kernel.reshape(output_channels, channels_col)

  result = newTensorUninit[T](batch_size, output_channels, output_height, output_width)
  var input_col = newTensorUninit[T](channels_col, output_height * output_width)
  var output: Tensor[T]

  for i in 0..<batch_size: #TODO: batch matmul
    im2col(input.atAxisIndex(0, i).squeeze(0), kernel_size, padding, stride, input_col)
    # The following must be done without copy: GEMM will directly write in the result tensor
    output = result.atAxisIndex(0, i).reshape(kernel_col.shape[0], input_col.shape[1])
    gemm(1.T, kernel_col, input_col, 0.T, output)

  if bias.rank > 0:
    result +.= bias.unsqueeze(0)

proc im2colgemm_conv2d_gradient*[T](input, kernel: Tensor[T],
                         padding: Size2D = (0,0),
                         stride: Size2D = (1,1),
                         grad_output: Tensor[T],
                         grad_input, grad_weight: var Tensor[T]) =
  ## Computes gradients w.r.t input and weights for a 2D convolution
  let
    batch_size = input.shape[^4]
    output_channels = kernel.shape[^4]
    kernel_size = (height: kernel.nchw_height, width: kernel.nchw_width)
    output_height = (input.nchw_height + (2*padding.height) - kernel.nchw_height) div stride.height + 1
    output_width = (input.nchw_width + (2*padding.width) - kernel.nchw_width) div stride.width + 1
    output_flatten_size = output_height*output_width
    channels_col = input.nchw_channels * kernel_size.height * kernel_size.width
    kernel_col = kernel.reshape(output_channels, input.nchw_channels*kernel.nchw_height*kernel.nchw_width)

  # Check if grad output shape looks correct
  assert grad_output.nchw_width == output_width and grad_output.nchw_height == output_height
  assert grad_output.nchw_channels == output_channels
  assert grad_output.shape[0] == input.shape[0]

  grad_input = zeros[T](batch_size, input.nchw_channels, input.nchw_height, input.nchw_width)
  grad_weight = zeros[T](output_channels, kernel.nchw_channels, kernel.nchw_height, kernel.nchw_width)
  var input_col = newTensorUninit[T](channels_col, output_height * output_width)

  for i in 0..<batch_size:
    let
      grad_output_col = grad_output.atAxisIndex(0, i).reshape(output_channels, output_flatten_size)
      grad_input_col = kernel_col.transpose() * grad_output_col

    im2col(input.atAxisIndex(0, i).squeeze(0), kernel_size, padding, stride, input_col)
    grad_input[i, _, _, _] = col2im(grad_input_col, input.nchw_channels, input.nchw_height, input.nchw_width, kernel_size, padding, stride).unsqueeze(0)
    grad_weight += (grad_output_col * input_col.transpose()).reshape(grad_weight.shape)
