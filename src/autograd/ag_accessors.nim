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

import ../tensor/tensor,
       ./ag_data_structure

template `[]`*[TT](v: Variable[TT], args: varargs[untyped]): Variable[TT] =
  ## Slice the tensor contained by the dynamic graph Variable
  ## Input:
  ##   - a Variable
  ## Output:
  ##   - a sliced Variable

  # Ensure that v is only called once even if it's a function with side-effects
  let z = v

  # TODO: backprop support
  var result: type(z)
  new result

  result.context = z.context
  result.value = z.value[args]
  result.grad = z.grad[args]
  result.requires_grad = z.requires_grad

  result

  # TODO: tests for slicing correspondence
