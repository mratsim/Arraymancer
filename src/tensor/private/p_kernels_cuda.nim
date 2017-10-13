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

import ./p_kernels_interface_cuda

include incl_kernels_cuda,
        incl_higher_order_cuda,
        incl_accessors_cuda

proc cuda_inPlaceAdd = discard # This is a hack so that the symbol is open
cuda_assign_glue(cuda_inPlaceAdd, "InPlaceAddOp")

proc cuda_inPlaceSub = discard # This is a hack so that the symbol is open
cuda_assign_glue(cuda_inPlaceSub, "InPlaceSubOp")

proc cuda_unsafeContiguous = discard # This is a hack so that the symbol is open
cuda_assign_glue(cuda_unsafeContiguous, "CopyOp")

proc cuda_Add = discard # This is a hack so that the symbol is open
cuda_binary_glue(cuda_Add, "AddOp")

proc cuda_Sub = discard # This is a hack so that the symbol is open
cuda_binary_glue(cuda_Sub, "SubOp")