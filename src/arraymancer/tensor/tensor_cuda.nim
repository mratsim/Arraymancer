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

import ./backend/cuda_global_state,
        ./backend/cuda,
        ./backend/cublas,
        ./init_cuda,
        ./init_copy_cuda,
        ./display_cuda,
        ./operators_blas_l1_cuda,
        ./operators_blas_l2l3_cuda,
        ./operators_broadcasted_cuda,
        ./shapeshifting_cuda

export  init_cuda,
        init_copy_cuda,
        display_cuda,
        operators_blas_l1_cuda,
        operators_blas_l2l3_cuda,
        operators_broadcasted_cuda,
        shapeshifting_cuda

# Override -std=-gnu++14
{.passC:"-std=c++14".}
