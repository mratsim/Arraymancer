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
        math

# Sigmoid cross-entropy function that works directly on Tensors
# and provide control without autograd

# Linear forward and backward
# TODO: layout version to accept both:
# - batch_first, NCHW (5D: NTCHW or NDCHW)
# - batch_last, CHWN (5D: CHWNT or CHWND) tensors.
proc linear*[T](x: var Tensor[T], weight: Tensor[T], bias: Tensor[T]) {.inline.} =
  x = weight * x
  x .+= bias

proc linear*[T](x: var Tensor[T], weight: Tensor[T]) {.inline.} =
  x = weight * x

proc linear_backward*[T](
        gradient: Tensor[T],
        cached_tensor,
        weight, bias: Tensor[T],
        dW, db: var Tensor[T]): Tensor[T] {.inline.} =
  result = weight.transpose * gradient
  gemm(gradient, cached_tensor.transpose, dW)

  db = sum(gradient, axis=0) # https://mlxai.github.io/2017/01/10/a-modular-approach-to-implementing-fully-connected-neural-networks.html

proc linear_backward*[T](
        gradient: Tensor[T],
        cached_tensor,
        weight: Tensor[T],
        dW: var Tensor[T]): Tensor[T] {.inline.} =
  result = weight.transpose * gradient
  gemm(gradient, cached_tensor.transpose, dW)

