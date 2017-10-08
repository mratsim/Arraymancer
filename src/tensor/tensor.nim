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

import sequtils, strutils, future, algorithm, nimblas, math, typetraits, macros, random

# Export OrderType (rowMajor, colMajor) from nimblas
export OrderType

import  ./backend/metadataArray,
        ./data_structure,
        ./init_cpu,
        ./accessors,
        ./accessors_macros_syntax,
        ./accessors_macros_read

export  metadataArray,
        data_structure,
        init_cpu,
        # ./init_deprecated_0_1_0,
        # ./init_cpu_deprecated_0_2_0, # source of deprecation spam https://github.com/nim-lang/Nim/issues/6436
        accessors,
        accessors_macros_syntax,
        accessors_macros_read

include # ./accessors_macros_read,
        ./accessors_macros_write,
        ./comparison,
        ./higher_order,
        ./higher_order_deprecated,
        ./shapeshifting,
        ./display,
        ./ufunc,
        ./operators_blas_l1,
        ./fallback/blas_l3_gemm,
        ./fallback/naive_l2_gemv,
        ./operators_blas_l2l3,
        ./operators_broadcasted,
        ./math_functions,
        ./filling_data,
        ./aggregate,
        ./term_rewriting,
        ./shortcuts,
        ./exporting


when defined(cuda):
  # Nimcuda poses issues with Nim docgen
  import nimcuda/[cuda_runtime_api, driver_types, cublas_api, cublas_v2, nimcuda]

when defined(cuda) or defined(doc):
  include ./backend/cuda_global_state,
          ./backend/cuda,
          ./backend/cublas,
          # ./backend/cublas_helper_proc,
          ./init_cuda,
          ./accessors_cuda,
          ./display_cuda,
          ./elementwise_cuda.nim,
          ./elementwise_glue_cuda.nim,
          ./higher_order_cuda,
          ./operators_blas_l1_cuda,
          ./operators_blas_l2l3_cuda,
          ./shapeshifting_cuda