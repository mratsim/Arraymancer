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


proc shape[T](s: openarray[T], dimensions: seq[int] = @[]): seq[int] {.noSideEffect.}=
    ## Helper function to get the dimensions of nested arrays/sequences
    ## C convention. Last index is the fastest changing (columns in 2D, depth in 3D) - Rows (slowest), Columns, Depth (fastest)
    # Dimension check is using only the first nested element so further checking
    # must be one to confirm that the total number of elements match the dimensions.
    result = dimensions & s.len
    when (T is seq|array):
      result = shape(s[0], result)

## Flatten any-depth nested sequences.
# TODO support array/openarray. Pending https://github.com/nim-lang/Nim/issues/2652
proc flatten[T](a: seq[T]): seq[T] {.noSideEffect.}= a
proc flatten[T](a: seq[seq[T]]): auto {.noSideEffect.}= a.concat.flatten