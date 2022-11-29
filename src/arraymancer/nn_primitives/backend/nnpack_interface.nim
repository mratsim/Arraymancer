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

import ../../tensor, ../private/p_nnp_types
import ./nnpack

proc nnpack_conv2d*(input, weight, bias: Tensor[float32], padding, stride: Size2D): Tensor[float32] {.noinit.}= # TODO use a single convention, return value or var result
  # Only accepts stride 1
  assert stride.width == 1 and stride.height == 1 # TODO CHange to a check + exception

  let
    batch_size = input.shape[0]
    output_channels = weight.shape[^4]
    output_height = (2*padding.height + input.nchw_height) - (weight.nchw_height - 1)
    output_width = (2*padding.width + input.nchw_width) - (weight.nchw_width - 1)

  # Make sure the data is contiguous before passing to nnpack
  let input = input.asContiguous()
  let weight = weight.asContiguous()
  var bias_nonnil: Tensor[float32] # TODO make bias truly optional and not just a tensor of rank 0


  # Bias with 0 rank means no bias at all
  if bias.rank == 0:
    # Temporary bias filled with zeros just to pass to nnpack
    bias_nonnil = zeros[float32](output_channels)
  else:
    bias_nonnil = bias.asContiguous()

  # Prepare tensor that the result will be stored on
  result = newTensorUninit[float32](input.shape[0], output_channels, output_height, output_width)

  let status = nnp_convolution_output(
    algorithm=nnp_convolution_algorithm_auto,
    batch_size=batch_size.csize_t,
    input_channels=input.nchw_channels.csize_t,
    output_channels=output_channels.csize_t,
    input_size=nnp_size(height: input.nchw_height.csize_t, width: input.nchw_width.csize_t),
    input_padding=nnp_padding(top: padding.height.csize_t, bottom: padding.height.csize_t,
                              left: padding.width.csize_t, right: padding.width.csize_t),
    kernel_size=nnp_size(height:weight.nchw_height.csize_t, width: weight.nchw_width.csize_t),
    input=cast[ptr cfloat](input.get_offset_ptr),
    kernel=cast[ptr cfloat](weight.get_offset_ptr),
    bias=cast[ptr cfloat](bias_nonnil.get_offset_ptr),
    output=cast[ptr cfloat](result.get_offset_ptr))
  assert status == nnp_status_success


proc nnpack_conv2d_gradient*[T](input, weight: Tensor[float32], padding, stride: Size2D,
                                grad_output: Tensor[T], grad_input, grad_weight: var Tensor[T]) =
  # Only accepts stride 1
  assert stride.width == 1 and stride.height == 1

  let
    batch_size = input.shape[0]
    input_channels = input.nchw_channels
    output_channels = weight.shape[^4]
    output_height = (2*padding.height + input.nchw_height) - (weight.nchw_height - 1)
    output_width = (2*padding.width + input.nchw_width) - (weight.nchw_width - 1)
    nninput_size = nnp_size(height: input.nchw_height.csize_t, width: input.nchw_width.csize_t)
    nnpadding = nnp_padding(top: padding.height.csize_t, bottom: padding.height.csize_t,
                              left: padding.width.csize_t, right: padding.width.csize_t)
    nnkernel_size = nnp_size(height:weight.nchw_height.csize_t, width: weight.nchw_width.csize_t)

  # Check if grad output shape looks correct
  assert grad_output.nchw_width == output_width and grad_output.nchw_height == output_height
  assert grad_output.nchw_channels == output_channels
  assert grad_output.shape[0] == input.shape[0]

  # Input gradient
  grad_input = zeros[T](input.shape)
  var status = nnp_convolution_input_gradient(
    algorithm=nnp_convolution_algorithm_auto,
    batch_size=batch_size.csize_t,
    input_channels=input_channels.csize_t,
    output_channels=output_channels.csize_t,
    input_size=nninput_size,
    input_padding=nnpadding,
    kernel_size=nnkernel_size,
    grad_output=cast[ptr cfloat](grad_output.get_offset_ptr),
    kernel=cast[ptr cfloat](weight.get_offset_ptr),
    grad_input=cast[ptr cfloat](grad_input.get_offset_ptr))
  assert status == nnp_status_success

  # Weight gradient
  grad_weight = zeros[T](weight.shape)
  status = nnp_convolution_kernel_gradient(
    algorithm=nnp_convolution_algorithm_auto,
    batch_size=batch_size.csize_t,
    input_channels=input_channels.csize_t,
    output_channels=output_channels.csize_t,
    input_size=nninput_size,
    input_padding=nnpadding,
    kernel_size=nnkernel_size,
    input=cast[ptr cfloat](input.get_offset_ptr),
    grad_output=cast[ptr cfloat](grad_output.get_offset_ptr),
    grad_kernel=cast[ptr cfloat](grad_weight.get_offset_ptr))
  assert status == nnp_status_success
