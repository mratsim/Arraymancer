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

import sequtils, strutils, future, algorithm, nimblas, math, typetraits, macros, random

# Export OrderType (rowMajor, colMajor) from nimblas
export OrderType

# include ../docs/autogen_nim_API
include arraymancer/utils/functional,
        arraymancer/utils/nested_containers,
        arraymancer/utils/ast_utils,
        arraymancer/backend/config_backends,
        arraymancer/global_config,
        arraymancer/backend/cuda_common,
        arraymancer/data_structure,
        arraymancer/init_cpu,
        arraymancer/init_deprecated,
        arraymancer/accessors,
        arraymancer/accessors_slicer,
        arraymancer/comparison,
        arraymancer/shapeshifting,
        arraymancer/display,
        arraymancer/higher_order,
        arraymancer/higher_order_deprecated,
        arraymancer/ufunc,
        arraymancer/operators_blas_l1,
        arraymancer/fallback/blas_l3_gemm,
        arraymancer/fallback/naive_l2_gemv,
        arraymancer/operators_blas_l2l3,
        arraymancer/operators_extra,
        arraymancer/aggregate,
        arraymancer/term_rewriting,
        arraymancer/exporting


# Unfortunately with that condition nimsuggest does not find included proc properly.
when defined(cuda):
  import nimcuda/[cuda_runtime_api, driver_types, cublas_api, cublas_v2, nimcuda]
  include ./arraymancer/backend/cuda_global_state,
          ./arraymancer/backend/cuda_specific,
          ./arraymancer/backend/cublas,
          ./arraymancer/init_cuda,
          ./arraymancer/accessors_cuda,
          ./arraymancer/display_cuda,
          ./arraymancer/functors_cuda.nim,
          ./arraymancer/functors_glue_cuda.nim,
          ./arraymancer/higher_order_cuda,
          ./arraymancer/operators_blas_l1_cuda,
          ./arraymancer/operators_blas_l2l3_cuda,
          ./arraymancer/shapeshifting_cuda