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

import ../src/arraymancer
import unittest


suite "Testing specific issues from bug tracker":
  test "Span slicing inside dynamic type procs fails to compile":
    # https://github.com/mratsim/Arraymancer/issues/43
    proc boo[T](): T =
      var a = zeros([2,2], int)
      echo a[1,_] #<-- Bug was undeclared identifier '_',
                  # unfortunately there is no way to gracefully check this
                  # with when not compiles for example

    # Check that our solution, export '_' doesn't create compatibility issue

    # tuple destructuring
    let (a, _, c) = (1, @[2,3],"hello")

    # https://github.com/mratsim/Arraymancer/issues/61
    proc foo[T](t: Tensor[T], x: int): Tensor[T] =
      t.unsafeSlice(x, _, _).unsafeReshape([t.shape[1], t.shape[2]])

    discard zeros([2,2,2], int).foo(1)