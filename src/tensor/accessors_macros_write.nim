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

import  ../private/ast_utils,
        ./private/p_accessors_macros_desugar,
        ./private/p_accessors_macros_write,
        ./data_structure,
        macros

macro `[]=`*[T](t: var Tensor[T], args: varargs[untyped]): untyped =
  ## Modifies a tensor inplace at the corresponding location or slice
  ##
  ##
  ## Input:
  ##   - a ``var`` tensor
  ##   - a location:
  ##     - specific coordinates (``varargs[int]``)
  ##     - or a slice (cf. tutorial)
  ##   - a value:
  ##     - a single value that will
  ##       - replace the value at the specific coordinates
  ##       - or be applied to the whole slice
  ##     - an openarray with a shape that matches the slice
  ##     - a tensor with a shape that matches the slice
  ## Result:
  ##   - Nothing, the tensor is modified in-place
  ## Usage:
  ##   - Assign a single value - foo[1..2, 3..4] = 999
  ##   - Assign an array/seq of values - foo[0..1,0..1] = [[111, 222], [333, 444]]
  ##   - Assign values from a view/Tensor - foo[^2..^1,2..4] = bar
  ##   - Assign values from the same Tensor - foo[^2..^1,2..4] = foo[^1..^2|-1, 4..2|-1]

  # varargs[untyped] consumes all arguments so the actual value should be popped
  # https://github.com/nim-lang/Nim/issues/5855

  var tmp = args
  let val = tmp.pop
  let new_args = getAST(desugar(tmp))

  result = quote do:
    slice_typed_dispatch_mut(`t`, `new_args`,`val`)


# # Linked to: https://github.com/mratsim/Arraymancer/issues/52
# Unfortunately enabling this breaksthe test suite
# "Setting a slice from a view of the same Tensor"

# macro `[]`*[T](t: var AnyTensor[T], args: varargs[untyped]): untyped =
#   ## Slice a Tensor or a CudaTensor
#   ## Input:
#   ##   - a Tensor or a CudaTensor
#   ##   - and:
#   ##     - specific coordinates (``varargs[int]``)
#   ##     - or a slice (cf. tutorial)
#   ## Returns:
#   ##   - a value or a tensor corresponding to the slice
#   ## Warning ⚠ CudaTensor temporary default:
#   ##   For CudaTensor only, this is a no-copy operation, data is shared with the input.
#   ##   This proc does not guarantee that a ``let`` value is immutable.
#   let new_args = getAST(desugar(args))

#   result = quote do:
#     slice_typed_dispatch_var(`t`, `new_args`)