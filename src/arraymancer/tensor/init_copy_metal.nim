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

import  ./private/p_init_metal,
        ./data_structure,
        ./backend/metal/metal_backend,
        ./init_metal,
        ./init_copy_cpu

proc clone*[T: SomeFloat](t: MetalTensor[T]): MetalTensor[T] {.noinit.} =
  ## Clone (deep copy) a MetalTensor.
  ## Copy will not share its data with the original.
  ## Data is copied via CPU to ensure proper handling of strides and layout.

  let cpuTensor = t.toCpu()
  let clonedCpu = init_copy_cpu.clone(cpuTensor)
  result = clonedCpu.metal()
