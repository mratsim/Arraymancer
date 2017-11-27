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
        ./higher_order

# Unfortunately higher_order depends on init_cpu and "clone" depends on higher_order, so we need an extra file
# to deal with circular dependencies

proc clone*[T](t: Tensor[T]): Tensor[T] {.noInit.}=
  ## Input:
  ##     - A tensor
  ## Returns:
  ##     - A full copy. The copy is recreated as a contiguous tensor.
  ##       If the input was a slice, unused values are discarded.
  result = t.map_inline(x)