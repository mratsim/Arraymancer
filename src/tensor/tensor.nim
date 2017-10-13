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

import nimblas
# Export OrderType (rowMajor, colMajor) from nimblas
export OrderType

import  ./backend/metadataArray,
        ./data_structure,
        ./init_cpu,
        # ./init_deprecated_0_1_0,
        # ./init_cpu_deprecated_0_2_0, # source of deprecation spam https://github.com/nim-lang/Nim/issues/6436
        ./accessors,
        ./accessors_macros_syntax,
        ./accessors_macros_read,
        ./accessors_macros_write,
        ./comparison,
        ./higher_order,
        # ./higher_order_deprecated,
        ./shapeshifting,
        ./display,
        ./ufunc,
        ./operators_blas_l1,
        ./operators_blas_l2l3,
        ./operators_broadcasted,
        ./math_functions,
        ./filling_data,
        ./aggregate,
        ./term_rewriting,
        ./shortcuts,
        ./exporting

export  metadataArray,
        data_structure,
        init_cpu,
        # init_deprecated_0_1_0,
        # init_cpu_deprecated_0_2_0, # source of deprecation spam https://github.com/nim-lang/Nim/issues/6436
        accessors,
        accessors_macros_syntax,
        accessors_macros_read,
        accessors_macros_write,
        comparison,
        higher_order,
        #higher_order_deprecated,
        shapeshifting,
        display,
        ufunc,
        operators_blas_l1,
        operators_blas_l2l3,
        operators_broadcasted,
        math_functions,
        filling_data,
        aggregate,
        term_rewriting,
        shortcuts,
        exporting

when defined(cuda):
  import ./tensor_cuda
  export tensor_cuda