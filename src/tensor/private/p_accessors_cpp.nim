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

# Unfortunately, mutable iterators do not work in cpp for
# primitives type. Bug report here: https://github.com/nim-lang/Nim/issues/4048

# As a workaround, and similar to what was done before to work around the lack of inline iterator chaining
# We return the memory index instead of a pointer to the value.
# Performance-wise it should be the same as what was done internally in the inline iterator
# must now be done by the proc calling the iterator.

import  ../data_structure,
        ./p_accessors,
        ./p_checks

iterator offsetValues*[T](t: Tensor[T]): (int, T) {.inline,noSideEffect.} =
  stridedIteration(IterKind.Offset_Values, t, 0, t.size)

iterator offsetValues*[T](t: Tensor[T], offset, size: int): (int, T) {.inline,noSideEffect.} =
  when compileOption("boundChecks"):
    check_contiguous_index(t, offset)
    check_contiguous_index(t, offset+size-1)
  stridedIteration(IterKind.Offset_Values, t, offset, size)

iterator zipOV*[T,U](t1: Tensor[T], t2: Tensor[U]): (int, T, U) {.inline,noSideEffect.} =
  when compileOption("boundChecks"):
    check_size(t1, t2)
  dualStridedIteration(IterKind.Offset_Values, t1, t2, 0, t1.size)

iterator zipOV*[T,U](t1: Tensor[T], t2: Tensor[U], offset, size: int): (int, T, U) {.inline,noSideEffect.} =
  when compileOption("boundChecks"):
    check_size(t1, t2)
    check_contiguous_index(t1, offset)
    check_contiguous_index(t1, offset+size-1)
  dualStridedIteration(IterKind.Offset_Values, t1, t2, offset, size)

iterator zipOV*[T,U,V](t1: Tensor[T], t2: Tensor[U], t3: Tensor[V]): (int, T, U, V) {.inline,noSideEffect.} =
  when compileOption("boundChecks"):
    check_size(t1, t2)
    check_size(t1, t3)
  tripleStridedIteration(IterKind.Offset_Values, t1, t2, t3, 0, t1.size)

iterator zipOV*[T,U,V]( t1: Tensor[T],
                      t2: Tensor[U],
                      t3: Tensor[V],
                      offset, size: int): (int, T, U, V) {.inline,noSideEffect.} =
  when compileOption("boundChecks"):
    check_size(t1, t2)
    check_size(t1, t3)
    check_contiguous_index(t1, offset)
    check_contiguous_index(t1, offset+size-1)
  tripleStridedIteration(IterKind.Offset_Values, t1, t2, t3, offset, size)