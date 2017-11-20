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

# Please compile with -d:cuda switch
import ../../src/arraymancer
import unittest, future

suite "CUDNN: Convolution 2D":
  test "Conv2d Forward":
    let input = [
      [1, 2, 0, 0],
      [5, 3, 0, 4],
      [0, 0, 0, 7],
      [9, 3, 0, 0]].toTensor().reshape(1,1,4,4).astype(float32).cuda
    let kernel = [
      [1, 1, 1],
      [1, 1, 0],
      [1, 0, 0]].toTensor().reshape(1,1,3,3).astype(float32).cuda
    let target = [
      [1,  8,  5,  0],
      [8, 11,  5,  4],
      [8, 17, 10, 11],
      [9, 12, 10,  7]].toTensor().reshape(1,1,4,4).astype(float32)
    let bias = [0].toTensor().reshape(1,1,1).astype(float32).cuda

    # TODO: padding should accept a tuple (i.e. unify Size2D and SizeHW)
    check: mean_absolute_error(
      input.conv2d(kernel, bias, padding=[1,1]).cpu,
      target) <= 1e-7'f32

  test "Conv2D Forward + Backward":

    let
      input = randomTensor([1,1,4,4], 1.0f).cuda
      kernel = randomTensor([1,1,3,3], 1.0f).cuda
      bias = randomTensor([1,1,1], 1.0f).cuda
      padding = [1,1]
      stride = [1,1]
      dilation = [1, 1]

    let output = conv2d(input, kernel, bias, padding, stride)

    let # Check gradient with float64 and cpu convolution
      dinput = input.cpu.astype(float)
      dkernel = kernel.cpu.astype(float)
      dbias = bias.cpu.astype(float)
      dpad = (1, 1) # TODO unify cudnn sizeHW and cpu size2D
      dstride = (1, 1)

    let
      target_grad_input = dinput.numerical_gradient(
        x => conv2d(x, dkernel, dbias, dpad, dstride).sum())
      target_grad_weight = dkernel.numerical_gradient(
        w => dinput.conv2d(w, dbias, dpad, dstride).sum())
      target_grad_bias = dbias.numerical_gradient(
        b => dinput.conv2d(dkernel, b, dpad, dstride).sum())

    var
      grad_input = zeros_like input
      grad_kernel = zeros_like kernel
      grad_bias = zeros_like bias

    let grad_output = ones_like(output)
    echo grad_output

    conv2d_backward(input, kernel, bias, padding, stride, dilation,
                    grad_output, grad_input, grad_kernel, grad_bias)