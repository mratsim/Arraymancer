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
        ./display

proc `$`*[T](t: CudaTensor[T]): string =
  ## Pretty-print a CudaTensor (when using ``echo`` for example)
  let desc = "Tensor of shape " & t.shape.join("x") & " of type \"" & T.name & "\" on backend \"" & "Cuda" & "\""
  
  let cpu_t = t.cpu()
  
  if t.rank <= 2:
    return desc & "\n" & cpu_t.disp2d
  elif t.rank == 3:
    return desc & "\n" & cpu_t.disp3d
  elif t.rank == 4:
    return desc & "\n" & cpu_t.disp4d
  else:
    return desc & "\n" & " -- NotImplemented: Display not implemented for tensors of rank > 4"