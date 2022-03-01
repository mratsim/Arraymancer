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

import  ./private/p_display,
        ./data_structure,
        typetraits

proc pretty*[T](t: Tensor[T], precision = -1): string =
  ## Pretty-print a Tensor with the option to set a custom `precision`
  ## for float values.
  var desc = t.type.name & " of shape \"" & $t.shape & "\" on backend \"" & "Cpu" & "\""
  if t.storage.isNil: # return useful message for uninit'd tensors instead of crashing
    return "Uninitialized " & $desc
  if t.size() == 0:
    return desc & "\n    [] (empty)"
  elif t.rank == 1: # for rank 1 we want an indentation, because we have no `|`
    return desc & "\n    " & t.prettyImpl(precision = precision)
  else:
    return desc & "\n" & t.prettyImpl(precision = precision)

proc `$`*[T](t: Tensor[T]): string =
  ## Pretty-print a tensor (when using ``echo`` for example)
  t.pretty()
