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
    let s = t.strides
    let (val,idx) = idx_data

    for i,j in s[0 .. ^2]: # We don't take the last element (the row)
        if idx mod j == 0:
            return $val & "|\n".repeat(s.high - i)
        if idx mod j == 1:
            return "|" & $val & "\t"
    return $val & "\t"

proc `$`*(t: Tensor): string {.noSideEffect.} =
    let indexed_data: seq[(string,int)] =
                      t.data.mapIt($it)
                            .zip(toSeq(1..t.strides[0])
                                 .cycle(t.shape[0])
                                )
    let str_tensor = indexed_data.foldl(a & t.bounds_display(b), "")
    let desc = "Tensor dimensions are " & t.shape.join("x")
    return str_tensor & "\n" & desc