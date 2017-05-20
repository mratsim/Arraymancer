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

import sequtils, strutils, future, algorithm, nimblas, math, typetraits, macros
include src/arraymancer/utils/functional,
        src/arraymancer/utils/nested_containers,
        src/arraymancer/utils/ast_utils,
        src/arraymancer/data_structure,
        src/arraymancer/accessors,
        src/arraymancer/accessors_slicer,
        src/arraymancer/comparison,
        src/arraymancer/display,
        src/arraymancer/init,
        src/arraymancer/ufunc,
        src/arraymancer/shapeshifting,
        src/arraymancer/blas