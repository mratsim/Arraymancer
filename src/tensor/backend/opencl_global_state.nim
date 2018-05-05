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

import nimcl, opencl

# ###################################################
# Global Cuda and CuBLAS state

# {.experimental.}

type clResource = PCommandQueue | PKernel | PProgram | PMem | PContext


# This was removed in master, feature request https://github.com/nim-lang/Nim/issues/7776
# proc `=destroy`*(clres: clResource) =
#   release clres

# TODO detect and use accelerators (FPGAs) or GPU by default
# And allow switching OpenCL device.
let (clDevice0*, clContext0*, clQueue0*) = singleDeviceDefaults()

