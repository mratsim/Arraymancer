# Copyright 2017-Present Mamy André-Ratsimbazafy & the Arraymancer contributors
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

import  ./arraymancer/tensor,
        ./arraymancer/nn_primitives/nn_primitives,
        ./arraymancer/autograd/autograd,
        ./arraymancer/nn/nn,
        ./arraymancer/nn_dsl/nn_dsl,
        ./arraymancer/datasets/mnist,
        ./arraymancer/datasets/imdb,
        ./arraymancer/io/io,
        ./arraymancer/ml/ml,
        ./arraymancer/stats/[stats, distributions, kde],
        ./arraymancer/nlp/nlp,
        ./arraymancer/tensor/einsum

export  tensor,
        nn_primitives,
        autograd,
        nn,
        nn_dsl,
        mnist,
        imdb,
        io,
        ml,
        stats, distributions, kde,
        nlp,
        einsum

when not defined(no_lapack):
  # The ml module also does not export everything is LAPACK is not available
  import ./arraymancer/linear_algebra/linear_algebra
  export linear_algebra
