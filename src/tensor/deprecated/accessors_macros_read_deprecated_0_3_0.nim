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


import  ../private/p_accessors_macros_desugar,
        ../private/p_accessors_macros_read,
        ../data_structure,
        macros

macro unsafeSlice*[T](t: Tensor[T], args: varargs[untyped]): untyped {.deprecated.}=
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
    unsafe_slice_typed_dispatch(`t`, `new_args`)