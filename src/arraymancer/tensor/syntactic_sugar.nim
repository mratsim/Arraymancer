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

template at_mut*[T](t: var Tensor[T], args: varargs[untyped]): untyped =
  ## Slice a Tensor, collapse singleton dimension, returning a mutable slice of the input
  ##
  ## This can be useful, for example, when assigning a value into a chain
  ## of slice operations which are usually considered immutable even if
  ## the original tensor is mutable. For example, this lets you do:
  ##
  ## .. code:: nim
  ##   var x = arange(12).reshape([4, 3])
  ##   let condition = [[true, false, true], [true, false, true]].toTensor
  ##   # The code `x[1..2, _][condition] = 1000` would fail with
  ##   # a `a slice of an immutable tensor cannot be assigned to` error
  ##   # Instead, using `at_mut` allows assignment to the slice
  ##   x.at_mut(1..2, _)[condition] = 1000
  ##
  ## Input:
  ##   - a Tensor
  ##   - and:
  ##     - specific coordinates (``varargs[int]``)
  ##     - or a slice (cf. tutorial)
  ## Returns:
  ##   - a mutable value or view of the Tensor corresponding to the slice
  ##     Singleton dimension are collapsed
  var mt = t[args].squeeze
  mt
