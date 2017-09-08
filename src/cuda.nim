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

import nimcuda/[cuda_runtime_api, driver_types, cublas_api, cublas_v2, nimcuda]

# arraymancer and arraymancer/cuda should not be both imported at the same time
# Unfortunately allowing this would require a difficult configuration to allow private proc visible to both modules
# but not exported externally
include ./arraymancer,
        ./arraymancer/init_cuda,
        ./arraymancer/display_cuda,
        ./arraymancer/operators_blas_l1_cuda,
        ./arraymancer/shapeshifting_cuda