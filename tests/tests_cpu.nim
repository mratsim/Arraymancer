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

{.push warning[Spacing]: off.}
import ../src/arraymancer,
        ./tensor/test_init,
        ./tensor/test_operators_comparison,
        ./tensor/test_accessors,
        ./tensor/test_accessors_slicer,
        ./tensor/test_display,
        ./tensor/test_operators_blas,
        ./tensor/test_math_functions,
        ./tensor/test_higherorder,
        ./tensor/test_aggregate,
        ./tensor/test_shapeshifting,
        ./tensor/test_broadcasting,
        ./tensor/test_ufunc,
        ./tensor/test_filling_data,
        ./tensor/test_optimization,
        ./tensor/test_exporting,
        ./io/test_csv,
        ./io/test_numpy,
        ./datasets/test_mnist,
        ./datasets/test_imdb,
        ./nn_primitives/test_nnp_numerical_gradient,
        ./nn_primitives/test_nnp_convolution,
        ./nn_primitives/test_nnp_loss,
        ./nn_primitives/test_nnp_maxpool,
        ./nn_primitives/test_nnp_gru,
        ./nn_primitives/test_nnp_embedding,
        ./autograd/test_gate_basic,
        ./autograd/test_gate_blas,
        ./autograd/test_gate_hadamard,
        ./autograd/test_gate_shapeshifting,
        ./ml/test_metrics,
        ./test_bugtracker

when not defined(no_lapack):
  import ./linear_algebra/test_linear_algebra,
        ./ml/test_dimensionality_reduction,
        ./ml/test_clustering

import  ./stability_tests/test_stability_openmp,
        # /end_to_end/examples_compile
        ./end_to_end/examples_run
{.pop.}
