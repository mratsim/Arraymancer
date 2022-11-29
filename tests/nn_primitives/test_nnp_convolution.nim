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


import ../../src/arraymancer
import unittest, sugar

proc main() =
  suite "Convolution 2D":
    block:
      let input = [
        [1, 2, 0, 0],
        [5, 3, 0, 4],
        [0, 0, 0, 7],
        [9, 3, 0, 0]].toTensor().reshape([1,1,4,4])
      let kernel = [
        [1, 1, 1],
        [1, 1, 0],
        [1, 0, 0]].toTensor().reshape([1,1,3,3])
      let target = [
        [1,  8,  5,  0],
        [8, 11,  5,  4],
        [8, 17, 10, 11],
        [9, 12, 10,  7]].toTensor().reshape([1,1,4,4])
      let bias = [0].toTensor().reshape(1,1,1)
      check: input.conv2d(kernel, bias, padding=(1,1)) == target

      let
        finput = input.astype(float32)
        fkernel = kernel.astype(float32)
        fbias = bias.astype(float32)
        ftarget = target.astype(float32)

      test "Simple Conv2D [Im2ColGEMM]":
        check: finput.conv2d(fkernel, fbias, padding=(1,1)).mean_absolute_error(ftarget) <= 1e-7'f32

      when defined(nnpack):
        test "Simple Conv2D [NNPack]":
          check: finput.conv2d(
            fkernel, fbias, padding=(1,1), algorithm=Conv2DAlgorithm.NNPackAuto
          ).mean_absolute_error(ftarget) <= 5e-6'f32 # TODO understand the loss of precision

    test "Strided Conv2D [Im2ColGEMM]":
      let input = [
        [
          [
            [2, 2, 0, 2, 1],
            [0, 1, 1, 0, 2],
            [1, 2, 1, 2, 1],
            [2, 2, 0, 0, 2],
            [2, 1, 1, 1, 2]
          ], [
            [2, 0, 1, 1, 1],
            [2, 2, 0, 0, 2],
            [2, 2, 1, 0, 0],
            [1, 1, 2, 2, 0],
            [2, 1, 1, 1, 0]
          ], [
            [0, 1, 2, 2, 0],
            [1, 1, 1, 1, 0],
            [2, 1, 2, 2, 0],
            [0, 2, 2, 2, 1],
            [0, 0, 2, 2, 1]
          ]
        ]].toTensor()

      let kernel =
        [
          [
            [
              [-1, -1, -1],
              [ 1,  0,  1],
              [ 0, -1,  0]
            ], [
              [ 1,  0, -1],
              [ 1, -1,  1],
              [ 0,  1,  0]
            ], [
              [ 0,  0,  1],
              [-1, -1, -1],
              [-1,  0,  0]
            ]
          ], [
            [
              [ 0,  1,  0],
              [ 1, -1, -1],
              [ 1,  1, -1]
            ], [
              [-1,  0,  1],
              [-1, -1,  1],
              [ 1,  1,  0]
            ], [
              [ 0,  1,  1],
              [-1,  1, -1],
              [-1, -1,  0]
            ]
          ]
        ].toTensor()

      let target =
        [
          [
            [ 2, -2,  0],
            [-3,  2, -5],
            [-2, -1,  0]
          ],[
            [-7,  1,  0],
            [ 3, -3,  2],
            [ 1,  3, -2]
          ]
        ].toTensor().reshape([1,2,3,3])

      let bias = [1,0].toTensor().reshape(2,1,1)

      check: input.conv2d(kernel, bias, padding=(1,1), stride=(2,2)) == target

      let
        finput = input.astype(float32)
        fkernel = kernel.astype(float32)
        fbias = bias.astype(float32)
        ftarget = target.astype(float32)

      check: finput.conv2d(fkernel, fbias, padding=(1,1), stride=(2,2)) == ftarget

    # Convolution 2d Forward + Backward
    block:
      let
        input = randomTensor([2,3,4,5], 1.0f)
        kernel = randomTensor([2,3,3,3], 1.0f)
        bias = randomTensor([2,1,1], 1.0f)
        padding = (1,1)
        stride = (1,1)

      let output = conv2d(input, kernel, bias, padding, stride)

      let
        dinput = input.astype(float)
        dkernel = kernel.astype(float)
        dbias = bias.astype(float)

      let
        target_grad_input = dinput.numerical_gradient(
          x => conv2d(x, dkernel, dbias, padding, stride).sum())
        target_grad_weight = dkernel.numerical_gradient(
          w => dinput.conv2d(w, dbias, padding, stride).sum())
        target_grad_bias = dbias.numerical_gradient(
          b => dinput.conv2d(dkernel, b, padding, stride).sum())

      var grad_input, grad_weight, grad_bias : Tensor[float32]
      let grad_output = ones[float32](output.shape)

      test "Conv2D Forward + Backward [Im2ColGEMM]":
        conv2d_backward(input, kernel, bias, padding, stride,
                        grad_output, grad_input, grad_weight, grad_bias)
        check: grad_bias.astype(float).mean_relative_error(target_grad_bias) < 1e-6
        check: grad_weight.astype(float).mean_relative_error(target_grad_weight) < 1e-6
        check: grad_input.astype(float).mean_relative_error(target_grad_input) < 1e-6

      when defined(nnpack):
        test "Conv2D Forward + Backward [NNPack]":
          conv2d_backward(input, kernel, bias, padding, stride,
                          grad_output, grad_input, grad_weight, grad_bias,
                          algorithm=Conv2DAlgorithm.NNPackAuto)
          check: grad_bias.astype(float).mean_relative_error(target_grad_bias) < 1e-6
          check: grad_weight.astype(float).mean_relative_error(target_grad_weight) < 1e-6
          check: grad_input.astype(float).mean_relative_error(target_grad_input) < 1e-6


main()
GC_fullCollect()
