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

import ../src/arraymancer
import unittest

proc main() =
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
      let (a{.used.}, _, c{.used.}) = (1, @[2,3],"hello")

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

    test "#307 3D slicing with same shape: offset is off":
      let x = zeros[int]([1, 2, 3])
      expect(IndexDefect):
        let y{.used.} = x[1, _, _]

    test "#386 Reshaping a contiguous permuted tensor":
      # https://github.com/mratsim/Arraymancer/issues/386
      block: # row-major
        let x = [[0, 1], [2, 3]].toTensor
        let expected = [[0, 2], [1, 3]].toTensor
        check:
          x.permute(1, 0) == expected
          x.permute(1, 0).reshape(2, 2) == expected
      block: # col-major
        let x = [[0, 1], [2, 3]].toTensor.clone(colMajor)
        let expected = [[0, 2], [1, 3]].toTensor
        check:
          x.permute(1, 0) == expected
          x.permute(1, 0).reshape(2, 2) == expected

    test "Test for PR #659 regression":
      ## Data taken from `kdtree` test case, in which the node `split`
      ## based on `data.percentile(50)` (i.e. `data.median`) suddenly
      ## changed after PR:
      ## https://github.com/mratsim/Arraymancer/pull/659/
      let t = @[@[0.195513 ,       0.225253],
                @[0.181441 ,       0.102758],
                @[0.26576  ,       0.218074],
                @[0.180852 ,     0.00262669],
                @[0.219789 ,       0.191867],
                @[0.160581 ,          0.131],
                @[0.269926 ,       0.237261],
                @[0.223423 ,       0.232116],
                @[0.191391 ,       0.183001],
                @[0.19654  ,      0.0809091],
                @[0.191497 ,      0.0929182],
                @[0.22709  ,       0.125705],
                @[0.263181 ,       0.124787],
                @[0.204926 ,     0.00688886],
                @[0.151998 ,      0.0531739],
                @[0.260266 ,      0.0583248],
                @[0.214864 ,       0.110506],
                @[0.247688 ,      0.0732228],
                @[0.246916 ,       0.204899],
                @[0.215206 ,       0.202225],
                @[0.242059 ,       0.102491],
                @[0.159926 ,       0.115765],
                @[0.249105 ,       0.200658],
                @[0.195783 ,       0.123984],
                @[0.17145  ,      0.0506388],
                @[0.258146 ,      0.0144846],
                @[0.215311 ,       0.222503],
                @[0.266231 ,       0.149363],
                @[0.178909 ,       0.142174],
                @[0.263406 ,      0.0867369],
                @[0.264824 ,       0.221786]
      ].toTensor

      const exp = 0.124787
      let arg = t[_, 1]
      check arg.offset == 1 # offset of 1 due to slicing
      # `reshape` copies here, because `arg` is a non contiguous tensor. Thus
      # offset must be reset to 0
      check arg.reshape([1, arg.size]).offset == 0
      check t[_, 1].percentile(50) == exp

main()
GC_fullCollect()
