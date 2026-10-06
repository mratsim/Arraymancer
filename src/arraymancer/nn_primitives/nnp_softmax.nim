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

import ../tensor

proc softmax*[T: SomeFloat](input: Tensor[T], axis: int = -1): Tensor[T] {.noinit.} =
  ## Softmax along an axis (default: -1, the last dimension).
  ## Numerically stable: exp(x - max(x)) / sum(exp(x - max(x)))
  let ax = if axis < 0: input.rank + axis else: axis
  when compileOption("boundChecks"):
    if ax < 0 or ax >= input.rank:
      raise newException(IndexDefect, "softmax axis " & $axis & " out of bounds for rank " & $input.rank)
  result = exp(input -. input.max(axis = ax))
  result /.= result.sum(axis = ax)

proc softmax_backward*[T: SomeFloat](gradOutput, cached_softmax: Tensor[T], axis: int = -1): Tensor[T] {.noinit.} =
  ## Backward pass for softmax: dX = Y * (dY - Σ dY*Y)
  let ax = if axis < 0: cached_softmax.rank + axis else: axis
  when compileOption("boundChecks"):
    if ax < 0 or ax >= cached_softmax.rank:
      raise newException(IndexDefect, "softmax_backward axis " & $axis & " out of bounds for rank " & $cached_softmax.rank)
  let dot = (gradOutput *. cached_softmax).sum(axis = ax)
  result = gradOutput -. dot
  result *.= cached_softmax
