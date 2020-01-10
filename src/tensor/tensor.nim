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

import  ../laser/dynamic_stack_arrays,
        ./data_structure,
        ./init_cpu,
        ./init_copy_cpu,
        ./accessors,
        ./accessors_macros_syntax,
        ./accessors_macros_read,
        ./accessors_macros_write,
        ./operators_comparison,
        ./higher_order_applymap,
        ./higher_order_foldreduce,
        ./shapeshifting,
        ./selectors,
        ./display,
        ./ufunc,
        ./operators_blas_l1,
        ./operators_blas_l2l3,
        ./operators_broadcasted,
        ./operators_logical,
        ./math_functions,
        ./filling_data,
        ./aggregate,
        ./lapack,
        ./optim_ops_fusion,
        ./syntactic_sugar,
        ./exporting

export  metadataArray,
        data_structure,
        init_cpu,
        init_copy_cpu,
        accessors,
        accessors_macros_syntax,
        accessors_macros_read,
        accessors_macros_write,
        operators_comparison,
        higher_order_applymap,
        higher_order_foldreduce,
        shapeshifting,
        selectors,
        display,
        ufunc,
        operators_blas_l1,
        operators_blas_l2l3,
        operators_broadcasted,
        operators_logical,
        math_functions,
        filling_data,
        aggregate,
        lapack,
        optim_ops_fusion,
        syntactic_sugar,
        exporting

when defined(cuda) or defined(nimdoc) or defined(nimsuggest):
  import ./tensor_cuda
  export tensor_cuda

when defined(opencl) or defined(nimdoc) or defined(nimsuggest):
  import ./tensor_opencl
  export tensor_opencl
