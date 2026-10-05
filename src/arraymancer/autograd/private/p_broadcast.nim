# Copyright 2026 the Arraymancer contributors
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

import ../../tensor

proc reduce_broadcast_dims*[T](grad: Tensor[T], target_shape: Metadata): Tensor[T] =
  ## Sum a broadcasted gradient back to `target_shape`.
  if grad.shape == target_shape:
    return grad
  result = grad
  while result.rank > target_shape.len:
    result = result.sum(axis = 0).squeeze(0)
  for i in 0 ..< target_shape.len:
    if target_shape[i] == 1 and result.shape[i] > 1:
      result = result.sum(axis = i)
  if result.shape != target_shape:
    result = result.reshape(target_shape)
