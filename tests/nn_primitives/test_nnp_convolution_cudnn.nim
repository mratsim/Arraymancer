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
import unittest, sugar

proc test_conv(T: typedesc) =
  test "Conv2d Forward [" & $T & ']':
    let input = [
      [1, 2, 0, 0],
      [5, 3, 0, 4],
      [0, 0, 0, 7],
      [9, 3, 0, 0]].toTensor().reshape(1,1,4,4).astype(T).cuda
    let kernel = [
      [1, 1, 1],
      [1, 1, 0],
      [1, 0, 0]].toTensor().reshape(1,1,3,3).astype(T).cuda
    let target = [
      [1,  8,  5,  0],
      [8, 11,  5,  4],
      [8, 17, 10, 11],
      [9, 12, 10,  7]].toTensor().reshape(1,1,4,4).astype(T)
    let bias = [0].toTensor().reshape(1,1,1).astype(T).cuda

    # TODO: padding should accept a tuple (i.e. unify Size2D and SizeHW)
    check: input.conv2d(kernel, bias, padding=[1,1]).cpu
             .mean_absolute_error(target) <= T(1e-7)

  test "Conv2D Forward + Backward [" & $T & ']':

    let # Note: cudnn backward works only with float32, it segfauts with float64
      input = randomTensor([10,3,4,5], T(1)).cuda
      kernel = randomTensor([16,3,3,3], T(1)).cuda
      bias = randomTensor([16,1,1], T(1)).cuda
      padding = [1,1]
      stride = [1,1]
      dilation = [1,1]

    let output = conv2d(input, kernel, bias, padding, stride)

    let # Check gradient with cpu convolution.
        # Note: codegen for numerical gradient float32 is not working properly
        # FFT/Winograd in float vs double may be quite different
      dinput = input.cpu.astype(float64)
      dkernel = kernel.cpu.astype(float64)
      dbias = bias.cpu.astype(float64)
      dpad = (1, 1) # TODO unify cudnn sizeHW and cpu size2D
      dstride = (1, 1)

    let
      target_grad_input = dinput.numerical_gradient(
        x => conv2d(x, dkernel, dbias, dpad, dstride).sum())
      target_grad_kernel = dkernel.numerical_gradient(
        w => dinput.conv2d(w, dbias, dpad, dstride).sum())
      target_grad_bias = dbias.numerical_gradient(
        b => dinput.conv2d(dkernel, b, dpad, dstride).sum())

    var
      grad_input = zeros_like input
      grad_kernel = zeros_like kernel
      grad_bias = zeros_like bias

    let grad_output = ones_like(output)

    conv2d_backward(input, kernel, bias, padding, stride, dilation,
                    grad_output, grad_input, grad_kernel, grad_bias)

    # There is a huge difference between CuDNN and im2col cpu results
    # Ideally we would need a Cuda numerical_gradient
    # In practice it is not relevant as we can use low precision (float16) without issue in deep learning

    check: grad_bias.cpu.mean_relative_error(target_grad_bias.astype(T)) < 1e-6
    check: grad_kernel.cpu.mean_relative_error(target_grad_kernel.astype(T)) < 0.2
    check: grad_input.cpu.mean_relative_error(target_grad_input.astype(T)) < 0.2

    # echo "output"
    # echo output
    # echo conv2d(dinput, dkernel, dbias, dpad, dstride)

    # echo "grad_kernel"
    # echo target_grad_kernel
    # echo grad_kernel

    # echo "grad_input"
    # echo target_grad_input
    # echo grad_input

testSuite "CUDNN: Convolution 2D":
  test_conv(float32)
  test_conv(float64)
