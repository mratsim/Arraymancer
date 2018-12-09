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

import  ../../tensor/tensor,
        ../../autograd/autograd,
        typetraits


type Loss*[TT] = ref object of Gate[TT]
  target*: TT

type SparseLoss*[TT; Idx: SomeNumber or byte or char or enum] = ref object of Gate[TT]
  target*: Tensor[Idx]
