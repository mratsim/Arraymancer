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

import ../../src/arraymancer
import unittest


suite "Testing specific issues from bug tracker":
  test "#43: Span slicing inside dynamic type procs fails to compile":
    # https://github.com/mratsim/Arraymancer/issues/43
    proc boo[T](): T {.used.}=
      var a = zeros[int]([2,2])
      echo a[1,_] #<-- Bug was undeclared identifier '_',
                  # unfortunately there is no way to gracefully check this
                  # with when not compiles for example

    # Check that our solution, export '_' doesn't create compatibility issue

    # tuple destructuring
    {.push hints: off.}  ## TODO replaced by XDeclaredButNotUsed when https://github.com/nim-lang/Nim/issues/4044
    let (a, _, c) = (1, @[2,3],"hello")
    {.pop.}

  test "#61 Unable to use operator `_` in this example":
    block:
      # https://github.com/mratsim/Arraymancer/issues/61
      proc foo[T](t: Tensor[T], x: int): Tensor[T] =
        t[x, _, _].reshape(t.shape[1], t.shape[2])

      discard zeros[int]([2,2,2]).foo(1)
    block:
      let t = zeros[int](2,2,2,2,2,2)
      proc foo[T](t: Tensor[T]): Tensor[T] =
        discard t[0..1|1, 0..<2|1, 0..^1|1, ^1..0|-1, _, 1] # doesnt work

      discard t.foo()
