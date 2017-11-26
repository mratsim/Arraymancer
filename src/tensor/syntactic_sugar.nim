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
        ./shapeshifting

template at*[T](t: Tensor[T], args: varargs[untyped]): untyped =
  ## Slice a Tensor and collapse singleton dimension.
  ##
  ## Input:
  ##   - a Tensor
  ##   - and:
  ##     - specific coordinates (``varargs[int]``)
  ##     - or a slice (cf. tutorial)
  ## Returns:
  ##   - a value or a view of the Tensor corresponding to the slice
  ##     Singleton dimension are collapsed
  ## Usage:
  ##   See the ``[]`` macro
  t[args].squeeze
