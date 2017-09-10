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


# Tools to manipulate deep nested containers

proc shape[T: not char](s: openarray[T], parent_shape: seq[int] = @[]): seq[int] {.noSideEffect.}=
  ## Helper function to get the shape of nested arrays/sequences
  ## C convention. Last index is the fastest changing (columns in 2D, depth in 3D) - Rows (slowest), Columns, Depth (fastest)
  ## The second argument "shape" is used for recursive call on nested arrays/sequences
  # Dimension check is using only the first nested element so further checking
  # must be one to confirm that the total number of elements match the shape.
  result = parent_shape & s.len
  when (T is seq|array):
    result = shape(s[0], result)

iterator flatIter(s: string): string {.noSideEffect.} =
  yield s

iterator flatIter[T: not char](s: openarray[T]): auto {.noSideEffect.}=
  ## Inline iterator on any-depth seq or array
  ## Returns values in order
  for item in s:
    when item is array|seq:
      for subitem in flatIter(item):
        yield subitem
    else:
      yield item


proc shape(s: string|seq[char], parent_shape: seq[int] = @[]): seq[int] {.noSideEffect.}=
  ## Handle char / string
  if parent_shape == @[]:
    return @[1]
  else: return parent_shape

