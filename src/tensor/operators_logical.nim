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

import  ./data_structure,
        ./higher_order_applymap,
        ./init_cpu,
        ./shapeshifting

# TODO tests

# #############################
# logical ops

proc `and`*(a, b: Tensor[bool]): Tensor[bool] {.noInit.} =
  ## Tensor element-wise boolean and.
  ##
  ## Returns:
  ##   - A tensor of boolean
  result = map2_inline(a, b, x and y)

proc `or`*(a, b: Tensor[bool]): Tensor[bool] {.noInit.} =
  ## Tensor element-wise boolean or.
  ##
  ## Returns:
  ##   - A tensor of boolean
  result = map2_inline(a, b, x or y)

proc `xor`*(a, b: Tensor[bool]): Tensor[bool] {.noInit.} =
  ## Tensor element-wise boolean xor.
  ##
  ## Returns:
  ##   - A tensor of boolean
  result = map2_inline(a, b, x xor y)

proc `not`*(a: Tensor[bool]): Tensor[bool] {.noInit.} =
  ## Tensor element-wise boolean and.
  ##
  ## Returns:
  ##   - A tensor of boolean
  result = map_inline(a, not x)