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

template getIndex[B: static[Backend], T](t: Tensor[B,T], idx: varargs[int]): ptr T =
    ## Convert [i, j, k, l ...] to the proper index.
    when compileOption("boundChecks"):
        if idx.len != t.rank:
            raise newException(IndexError, "Number of arguments: " &
                                            $(idx.len) &
                                            ", is different from tensor rank: " &
                                            $(t.rank))
    var real_idx = t.offset
    ptrMath:
        for i,j in zip(t.strides,idx):
            real_idx += i*j
        when compileOption("boundChecks"):
            if real_idx >= (t.offset + t.data.len):
                raise newException(IndexError, "Index out of bounds")
    real_idx

proc `[]`*[B: static[Backend], T](t: Tensor[B,T], idx: varargs[int]): T {.noSideEffect.} =
    ## Get the value at input coordinates
    return t.getIndex(idx)[]

proc `[]=`*[B: static[Backend], T](t: var Tensor[B,T], idx: varargs[int], val: T) {.noSideEffect.} =
    ## Set the value at input coordinates
    t.getIndex(idx)[] = val