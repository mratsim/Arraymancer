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

proc `$`*[T](t: Tensor[T]): string =
  ## Pretty-print a tensor (when using ``echo`` for example)
  let desc = t.type.name & " of shape " & $t.shape & "\" on backend \"" & "Cpu" & "\""
  if t.rank <= 2:
    return desc & "\n" & t.disp2d
  elif t.rank == 3:
    return desc & "\n" & t.disp3d
  elif t.rank == 4:
    return desc & "\n" & t.disp4d
  else:
    return desc & "\n" & " -- NotImplemented: Display not implemented for tensors of rank > 4"
