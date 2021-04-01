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

import  ./laser/dynamic_stack_arrays,
        ./laser/strided_iteration/foreach,
        ./tensor/data_structure,
        ./tensor/init_cpu,
        ./tensor/init_copy_cpu,
        ./tensor/accessors,
        ./tensor/accessors_macros_syntax,
        ./tensor/accessors_macros_read,
        ./tensor/accessors_macros_write,
        ./tensor/operators_comparison,
        ./tensor/higher_order_applymap,
        ./tensor/higher_order_foldreduce,
        ./tensor/shapeshifting,
        ./tensor/selectors,
        ./tensor/display,
        ./tensor/ufunc,
        ./tensor/operators_blas_l1,
        ./tensor/operators_blas_l2l3,
        ./tensor/operators_broadcasted,
        ./tensor/operators_logical,
        ./tensor/math_functions,
        ./tensor/filling_data,
        ./tensor/aggregate,
        ./tensor/algorithms,
        ./tensor/lapack,
        ./tensor/optim_ops_fusion,
        ./tensor/syntactic_sugar,
        ./tensor/exporting

export  dynamic_stack_arrays,
        foreach,
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
        aggregate,
        algorithms,
        lapack,
        optim_ops_fusion,
        syntactic_sugar,
        exporting

when defined(cuda) or defined(nimdoc) or defined(nimsuggest):
  import ./tensor/tensor_cuda
  export tensor_cuda

when defined(opencl) or defined(nimdoc) or defined(nimsuggest):
  import ./tensor/tensor_opencl
  export tensor_opencl
