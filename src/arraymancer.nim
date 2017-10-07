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

when defined(doc):
  include ../docs/autogen_nim_API

import sequtils, strutils, future, algorithm, nimblas, math, typetraits, macros, random

# Export OrderType (rowMajor, colMajor) from nimblas
export OrderType

# include ../docs/autogen_nim_API
include arraymancer/utils/functional,
        arraymancer/utils/nested_containers,
        arraymancer/utils/ast_utils,
        arraymancer/global_config,
        arraymancer/backend/metadataArray,
        arraymancer/backend/blis,
        arraymancer/backend/openmp,
        arraymancer/data_structure,
        arraymancer/data_structure_helpers,
        arraymancer/init_cpu,
        arraymancer/init_deprecated_0_1_0,
        arraymancer/init_cpu_deprecated_0_2_0, # source of deprecation spam https://github.com/nim-lang/Nim/issues/6436
        arraymancer/accessors,
        arraymancer/accessors_macros_syntax,
        arraymancer/accessors_macros_desugar,
        arraymancer/accessors_macros_read,
        arraymancer/accessors_macros_write,
        arraymancer/comparison,
        arraymancer/higher_order,
        arraymancer/higher_order_deprecated,
        arraymancer/shapeshifting,
        arraymancer/display,
        arraymancer/ufunc,
        arraymancer/operators_blas_l1,
        arraymancer/fallback/blas_l3_gemm,
        arraymancer/fallback/naive_l2_gemv,
        arraymancer/operators_blas_l2l3,
        arraymancer/operators_broadcasted,
        arraymancer/math_functions,
        arraymancer/filling_data,
        arraymancer/aggregate,
        arraymancer/term_rewriting,
        arraymancer/shortcuts,
        arraymancer/exporting


when defined(cuda):
  # Nimcuda poses issues with Nim docgen
  import nimcuda/[cuda_runtime_api, driver_types, cublas_api, cublas_v2, nimcuda]

when defined(cuda) or defined(doc):
  include ./arraymancer/backend/cuda_global_state,
          ./arraymancer/backend/cuda,
          ./arraymancer/backend/cublas,
          # ./arraymancer/backend/cublas_helper_proc,
          ./arraymancer/init_cuda,
          ./arraymancer/accessors_cuda,
          ./arraymancer/display_cuda,
          ./arraymancer/elementwise_cuda.nim,
          ./arraymancer/elementwise_glue_cuda.nim,
          ./arraymancer/higher_order_cuda,
          ./arraymancer/operators_blas_l1_cuda,
          ./arraymancer/operators_blas_l2l3_cuda,
          ./arraymancer/shapeshifting_cuda