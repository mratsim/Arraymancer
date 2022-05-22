# Copyright 2017-2020 the Arraymancer contributors
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

import
  macros,
  ../data_structure

# Experimental supports for empty tensors
# ------------------------------------------------------------------------
# If there is a spec for Numpy tensor, please add to
# - https://github.com/mratsim/Arraymancer/issues/210
# - or https://github.com/mratsim/Arraymancer/issues/445

template isEmpty[T](oa: openarray[T]): bool =
  oa.len == 0

template isEmpty(t: AnyTensor): bool =
  size(t) == 0

template skipIfEmpty*(t: typed): untyped =
  ## Skip the iteration of a "for-loop" or "while-loop"
  ## if the tensor is empty
  if isEmpty(t):
    continue

macro returnEmptyIfEmpty*(tensors: varargs[untyped]): untyped =
  ## If any of the argument tensors are empty
  ## return an empty tensor
  result = newStmtList()
  for tensor in tensors: # static for loop
    let isEmptyCall = newCall(bindSym"isEmpty", tensor)
    result.add quote do:
      if `isEmptyCall`:
        return newTensor[getSubType(type(result))](0)
