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

import  ../../private/functional,
        ../data_structure

proc check_nested_elements*(shape: seq[int], len: int) {.noSideEffect, inline.}=
  ## Compare the detected shape from flatten with the real length of the data
  ## Input:
  ##   -- A shape (sequence of int)
  ##   -- A length (int)
  if (shape.product != len):
    raise newException(IndexError, "Each nested sequence at the same level must have the same number of elements")

proc check_index*(t: Tensor, idx: varargs[int]) {.noSideEffect.}=
  if idx.len != t.rank:
    raise newException(IndexError, "Number of arguments: " &
                    $(idx.len) &
                    ", is different from tensor rank: " &
                    $(t.rank))

proc check_elementwise*(a, b:AnyTensor)  {.noSideEffect.}=
  ## Check if element-wise operations can be applied to 2 Tensors
  if a.shape != b.shape:
    raise newException(ValueError, "Both Tensors should have the same shape.\n Left-hand side has shape " &
                                   $a.shape & " while right-hand side has shape " & $b.shape)

proc check_size*[T,U](a:Tensor[T], b:Tensor[U])  {.noSideEffect.}=
  ## Check if the total number of elements match

  if a.size != b.size:
    raise newException(ValueError, "Both Tensors should have the same total number of elements.\n" &
      "Left-hand side has " & $a.size & " (shape: " & $a.shape & ") while right-hand side has " &
      $b.size & " (shape: " & $b.shape & ")."
    )