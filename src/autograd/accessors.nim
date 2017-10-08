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
       ./data_structure

template `[]`*[TT](v: Variable[TT], args: varargs[untyped]): Variable[TT] =
  var result: type(v)
  new result

  result.tape = v.tape
  result.ancestor = v.ancestor
  result.value = v.value.unsafeSlice(args)
  result.grad = v.grad.unsafeSlice(args)

  result

  # TODO: tests for slicing correspondance