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

# Nvidia CuDNN backend configuration
# Note: Having CUDA installed does not mean CuDNN is installed

import nimcuda/[nimcuda, cudnn]

export nimcuda, cudnn


var defaultHandle_cudnn*: cudnnHandle_t
check cudnnCreate(addr defaultHandle_cudnn)

proc cudnnRelease() {.noconv.} =
  # Release CuDNN resources
  check cudnnDestroy(defaultHandle_cudnn)

  when defined(debug):
    echo "CuDNN resources successfully released"

addQuitProc(cudnnRelease)

