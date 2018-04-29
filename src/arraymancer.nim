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

import  ./tensor/tensor,
        ./nn_primitives/nn_primitives,
        ./autograd/autograd,
        ./nn/nn,
        ./nn_dsl/nn_dsl,
        ./datasets/mnist,
        ./io/io_csv,
        ./ml/ml

export  tensor,
        nn_primitives,
        autograd,
        nn,
        nn_dsl,
        mnist,
        io_csv,
        ml

when not defined(no_lapack):
  import ./linear_algebra/linear_algebra
  export linear_algebra
