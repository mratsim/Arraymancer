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

import  ./private/p_accessors_macros_desugar,
        ./private/p_accessors_macros_read,
        ./data_structure,
        macros

macro `[]`*[T](t: AnyTensor[T], args: varargs[untyped]): untyped =
  ## Slice a Tensor or a CudaTensor
  ## Input:
  ##   - a Tensor or a CudaTensor
  ##   - and:
  ##     - specific coordinates (``varargs[int]``)
  ##     - or a slice (cf. tutorial)
  ## Returns:
  ##   - a value or a tensor corresponding to the slice
  ## Warning ⚠ CudaTensor temporary default:
  ##   For CudaTensor only, this is a no-copy operation, data is shared with the input.
  ##   This proc does not guarantee that a ``let`` value is immutable.
  ## Usage:
  ##    - Basic indexing - foo[2, 3]
  ##    - Basic indexing - foo[1+1, 2*2*1]
  ##    - Basic slicing - foo[1..2, 3]
  ##    - Basic slicing - foo[1+1..4, 3-2..2]
  ##    - Span slices - foo[_, 3]
  ##    - Span slices - foo[1.._, 3]
  ##    - Span slices - foo[_..3, 3]
  ##    - Span slices - foo[_.._, 3]
  ##    - Stepping - foo[1..3\|2, 3]
  ##    - Span stepping - foo[_.._\|2, 3]
  ##    - Span stepping - foo[_.._\|+2, 3]
  ##    - Span stepping - foo[1.._\|1, 2..3]
  ##    - Span stepping - foo[_..<4\|2, 3]
  ##    - Slicing until at n from the end - foo[0..^4, 3]
  ##    - Span Slicing until at n from the end - foo[_..^2, 3]
  ##    - Stepped Slicing until at n from the end - foo[1..^1\|2, 3]
  ##    - Slice from the end - foo[^1..0\|-1, 3]
  ##    - Slice from the end - expect non-negative step error - foo[^1..0, 3]
  ##    - Slice from the end - foo[^(2*2)..2*2, 3]
  ##    - Slice from the end - foo[^3..^2, 3]
  let new_args = getAST(desugar(args))

  result = quote do:
    inner_typed_dispatch(`t`, `new_args`)

macro unsafeSlice*[T](t: Tensor[T], args: varargs[untyped]): untyped =
  ## Slice a Tensor
  ## Input:
  ##   - a Tensor
  ##   - and:
  ##     - specific coordinates (``varargs[int]``)
  ##     - or a slice (cf. tutorial)
  ## Returns:
  ##   - a value or a view of the Tensor corresponding to the slice
  ## Warning ⚠:
  ##   This is a no-copy operation, data is shared with the input.
  ##   This proc does not guarantee that a ``let`` value is immutable.
  ## Usage:
  ##   See the ``[]`` macro
  let new_args = getAST(desugar(args))

  result = quote do:
    unsafe_inner_typed_dispatch(`t`, `new_args`)