# Copyright 2017 Mamy André-Ratsimbazafy
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

import ./data_structure, ./accessors,
        ./higher_order_applymap,
        ./init_cpu,
        ./shapeshifting

proc `==`*[T](a,b: Tensor[T]): bool {.noSideEffect.}=
  ## Tensor comparison
  if a.shape != b.shape: return false

  for a, b in zip(a,b):
    ## Iterate through the tensors using stride-aware iterators
    ## Returns early if false
    if a != b: return false
  return true

# TODO tests

# ###############
# broadcasted ops

proc `.==`*[T](a, b: Tensor[T]): Tensor[bool] {.noInit.} =
  ## Tensor element-wise equality.
  ##
  ## And broadcasted element-wise equality.
  ##
  ## Returns:
  ##   - A tensor of boolean
  let (tmp_a, tmp_b) = broadcast2(a, b)
  result = map2_inline(tmp_a, tmp_b, x == y)

proc `.!=`*[T](a, b: Tensor[T]): Tensor[bool] {.noInit.} =
  ## Tensor element-wise inequality.
  ##
  ## And broadcasted element-wise inequality.
  ##
  ## Returns:
  ##   - A tensor of boolean
  let (tmp_a, tmp_b) = broadcast2(a, b)
  result = map2_inline(tmp_a, tmp_b, x != y)

proc `.<=`*[T](a, b: Tensor[T]): Tensor[bool] {.noInit.} =
  ## Tensor element-wise lesser or equal.
  ##
  ## And broadcasted element-wise lesser or equal.
  ##
  ## Returns:
  ##   - A tensor of boolean
  let (tmp_a, tmp_b) = broadcast2(a, b)
  result = map2_inline(tmp_a, tmp_b, x <= y)

proc `.<`*[T](a, b: Tensor[T]): Tensor[bool] {.noInit.} =
  ## Tensor element-wise lesser than.
  ##
  ## And broadcasted element-wise lesser than.
  ##
  ## Returns:
  ##   - A tensor of boolean
  let (tmp_a, tmp_b) = broadcast2(a, b)
  result = map2_inline(tmp_a, tmp_b, x < y)

proc `.>=`*[T](a, b: Tensor[T]): Tensor[bool] {.noInit.} =
  ## Tensor element-wise greater or equal.
  ##
  ## And broadcasted element-wise greater or equal.
  ##
  ## Returns:
  ##   - A tensor of boolean
  let (tmp_a, tmp_b) = broadcast2(a, b)
  result = map2_inline(tmp_a, tmp_b, x >= y)

proc `.>`*[T](a, b: Tensor[T]): Tensor[bool] {.noInit.} =
  ## Tensor element-wise greater than.
  ##
  ## And broadcasted element-wise greater than.
  ##
  ## Returns:
  ##   - A tensor of boolean
  let (tmp_a, tmp_b) = broadcast2(a, b)
  result = map2_inline(tmp_a, tmp_b, x > y)