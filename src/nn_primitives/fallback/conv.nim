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

import  ../../tensor/tensor, ../types

proc im2col[T](input: Tensor[T], kernel_size: Size2D,
               padding: Size2D = (0,0), stride: Size2D = (1,1)): Tensor[T] =
  ## Convert blocks of an image into columns, useful for preprocessing
  ## an image before convolutions
  let
    channels = input.nchw_channels
    height = input.nchw_height
    width = input.nchw_width
    channels_col = channels * kernel_size.height * kernel_size.width
    height_col = (height + (2 * padding.height) - kernel_size.height) div stride.height + 1
    width_col = (width + (2 * padding.width) - kernel_size.width) div stride.width + 1
  result = newTensorUninit[T](channels_col, height_col * width_col)
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
          result[c, offset_col + w] = 0
        else:
          result[c, offset_col + w] = input[c_offset, row, col]

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
    kernel_col = kernel.unsafeReshape(output_channels, input.nchw_channels*kernel.nchw_height*kernel.nchw_width)

  result = newTensorUninit[T](batch_size, output_channels, output_height, output_width)

  for i in 0..<batch_size:
    let input_col = im2col(input.unsafeAtAxisIndex(0, i).unsafeSqueeze(0), kernel_size, padding, stride)
    var output = result.unsafeAtAxisIndex(0, i).unsafeReshape(kernel_col.shape[0], input_col.shape[1])
    gemm(kernel_col, input_col, output)

  if bias.rank > 0:
    result .+= bias.unsafeUnsqueeze(0)

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
    kernel_col = kernel.unsafeReshape(output_channels, input.nchw_channels*kernel.nchw_height*kernel.nchw_width)

  # Check if grad output shape looks correct
  assert grad_output.nchw_width == output_width and grad_output.nchw_height == output_height
  assert grad_output.nchw_channels == output_channels
  assert grad_output.shape[0] == input.shape[0]

  grad_input = zeros[T](batch_size, input.nchw_channels, input.nchw_height, input.nchw_width)
  grad_weight = zeros[T](output_channels, kernel.nchw_channels, kernel.nchw_height, kernel.nchw_width)

  for i in 0..<batch_size:
    let
      input_col = im2col(input.unsafeAtAxisIndex(0, i).unsafeSqueeze(0), kernel_size, padding, stride)
      grad_output_col = grad_output.unsafeAtAxisIndex(0, i).unsafeReshape(output_channels, output_flatten_size)
      grad_input_col = kernel_col.unsafeTranspose() * grad_output_col
    grad_input[i, _, _, _] = col2im(grad_input_col, input.nchw_channels, input.nchw_height, input.nchw_width, kernel_size, padding, stride).unsafeUnsqueeze(0)
    grad_weight += (grad_output_col * input_col.unsafeTranspose()).unsafeReshape(grad_weight.shape)
