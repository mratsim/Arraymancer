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
        ./arraymancer/nn_primitives,
        ./arraymancer/autograd,
        ./arraymancer/nn,
        ./arraymancer/datasets/mnist,
        ./arraymancer/datasets/imdb,
        ./arraymancer/io,
        ./arraymancer/ml,
        ./arraymancer/stats/[stats, distributions, kde],
        ./arraymancer/spatial/kdtree,
        ./arraymancer/nlp,
        ./arraymancer/tensor/einsum,
        ./arraymancer/linear_algebra,
        ./arraymancer/std_version_types

export  tensor,
        nn_primitives,
        autograd,
        nn,
        mnist,
        imdb,
        io,
        ml,
        stats, distributions, kde,
        kdtree,
        nlp,
        einsum,
        std_version_types

when not defined(no_lapack):
  # The ml module also does not export everything is LAPACK is not available
  import ./arraymancer/linear_algebra
  export linear_algebra
