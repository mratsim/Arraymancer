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

# Please compile with -d:cuda switch
{.push warning[Spacing]: off.}
import ../src/arraymancer,
        ./tensor/test_init_cuda,
        ./tensor/test_operators_blas_cuda,
        ./tensor/test_accessors_slicer_cuda,
        ./tensor/test_shapeshifting_cuda,
        ./tensor/test_broadcasting_cuda

# Please compile with -d:cudnn switch
when not defined(cudnn):
  echo "CuDNN tests skipped, please pass -d:cudnn flag if you want to enable cudnn tests after cuda tests."
else:
  import ./nn_primitives/test_nnp_convolution_cudnn
{.pop.}
