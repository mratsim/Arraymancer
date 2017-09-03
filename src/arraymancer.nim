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

import sequtils, strutils, future, algorithm, nimblas, math, typetraits, macros, random

include ../docs/autogen_nim_API
include arraymancer/utils/functional,
        arraymancer/utils/nested_containers,
        arraymancer/utils/ast_utils,
        arraymancer/backend/config_backends,
        arraymancer/data_structure,
        arraymancer/init,
        arraymancer/accessors,
        arraymancer/accessors_slicer,
        arraymancer/comparison,
        arraymancer/shapeshifting,
        arraymancer/display,
        arraymancer/ufunc,
        arraymancer/fallback/blas_l3_gemm,
        arraymancer/operators_blas,
        arraymancer/operators_extra,
        arraymancer/aggregate,
        arraymancer/term_rewriting