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

proc bounds_display(t: Tensor,
                        idx_data: tuple[val: string,
                                  idx: int]
                        ): string  {.noSideEffect.} =
    ## Internal routine, compare an index with the strides of a Tensor
    ## to check beginning and end of lines
    ## Add the delimiter "|" and line breaks at beginning and end of lines
    let s = t.strides
    let (val,idx) = idx_data

    for i,j in s[0 .. ^2]: # We don't take the last element (the row)
        if idx mod j == 0:
            return $val & "|\n".repeat(s.high - i)
        if idx mod j == 1:
            return "|" & $val & "\t"
    return $val & "\t"

proc `$`*[B,T](t: Tensor[B,T]): string {.noSideEffect.} =
    ## Display a tensor

    # Add a position index to each value in the Tensor
    let indexed_data: seq[(string,int)] =
                      t.data.mapIt($it)
                            .zip(toSeq(1..t.strides[0])
                                 .cycle(t.dimensions[0]+1)
                                )
    
    # Create a closure to apply the boundaries transformation for the specific input
    proc curry_bounds(tup: (string,int)): string {.noSideEffect.} = t.bounds_display(tup)

    let str_tensor = indexed_data.concatMap(curry_bounds)
    let desc = "Tensor of shape " & t.shape.join("x") & " of type \"" & T.name & "\" on backend \"" & $B & "\""
    return desc & "\n" & str_tensor