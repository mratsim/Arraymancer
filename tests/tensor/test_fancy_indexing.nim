# Copyright 2017-2020 Mamy André-Ratsimbazafy
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

import ../../src/arraymancer, ../testutils
import unittest

proc main() =
  suite "Fancy indexing":
    # x = np.array([[ 4, 99,  2],
    #               [ 3,  4, 99],
    #               [ 1,  8,  7],
    #               [ 8,  6,  8]])

    let x = [[ 4, 99,  2],
            [ 3,  4, 99],
            [ 1,  8,  7],
            [ 8,  6,  8]].toTensor()

    test "Index selection via fancy indexing":
      block: # print(x[:, [0, 2]])
        let r = x[_, [0, 2]]

        let exp = [[ 4,  2],
                  [ 3, 99],
                  [ 1,  7],
                  [ 8,  8]].toTensor()

        check: r == exp

      block: # print(x[[1, 3], :])
        let r = x[[1, 3], _]

        let exp = [[3, 4, 99],
                  [8, 6,  8]].toTensor()

        check: r == exp

    test "Masked selection via fancy indexing":
      block:
        let r = x[x >. 50]
        let exp = [99, 99].toTensor()
        check: r == exp

      block:
        let r = x[x <. 50]
        let exp = [4, 2, 3, 4, 1, 8, 7, 8, 6, 8].toTensor()
        check: r == exp

    test "Masked axis selection via fancy indexing":
      block: # print('x[:, np.sum(x, axis = 0) > 50]')
        let r = x[_, x.sum(axis = 0) >. 50]

        let exp = [[99, 2],
                  [ 4, 99],
                  [ 8, 7],
                  [ 6, 8]].toTensor()

        check: r == exp

      block: # print('x[np.sum(x, axis = 1) > 50, :]')
        let r = x[x.sum(axis = 1) >. 50, _]

        let exp = [[4, 99, 2],
                  [3, 4, 99]].toTensor()

        check: r == exp

    test "Index assign value via fancy indexing":
      block: # y[:, [0, 2]] = -100
        var y = x.clone()
        y[_, [0, 2]] = -100

        let exp = [[-100, 99, -100],
                  [-100,  4, -100],
                  [-100,  8, -100],
                  [-100,  6, -100]].toTensor()

        check: y == exp

      block: # y[[1, 3], :] = -100
        var y = x.clone()
        y[[1, 3], _] = -100

        let exp = [[   4,   99,    2],
                  [-100, -100, -100],
                  [   1,    8,    7],
                  [-100, -100, -100]].toTensor()

        check: y == exp

    test "Masked assign value via fancy indexing":
      block: # y[y > 50] = -100
        var y = x.clone()
        y[y >. 50] = -100

        let exp = [[ 4, -100,    2],
                  [ 3,    4, -100],
                  [ 1,    8,    7],
                  [ 8,    6,    8]].toTensor()

        check: y == exp

      block: # y[y < 50] = -100
        var y = x.clone()
        y[y <. 50] = -100

        let exp = [[ -100,   99, -100],
                  [ -100, -100,   99],
                  [ -100, -100, -100],
                  [ -100, -100, -100]].toTensor()

        check: y == exp

    test "Masked axis assign value via fancy indexing":
      block: # y[:, y.sum(axis = 0) > 50] = -100
        var y = x.clone()
        y[_, y.sum(axis = 0) >. 50] = -100

        let exp = [[  4, -100, -100],
                  [  3, -100, -100],
                  [  1, -100, -100],
                  [  8, -100, -100]].toTensor()

        check: y == exp

      block: # y[y.sum(axis = 1) > 50, :] = -100
        var y = x.clone()
        y[y.sum(axis = 1) >. 50, _] = -100

        let exp = [[-100, -100, -100],
                  [-100, -100, -100],
                  [   1,    8,    7],
                  [   8,    6,    8]].toTensor()

        check: y == exp

    test "Masked axis assign tensor via fancy indexing - invalid Numpy syntaxes":
      block: # y[:, y.sum(axis = 0) > 50] = np.array([10, 20, 30, 40])
        var y = x.clone()

        expect(IndexError):
          y[_, y.sum(axis = 0) >. 50] = [10, 20, 30, 40].toTensor()

    test "Masked axis assign broadcastable 1d tensor via fancy indexing":
      block: # y[:, y.sum(axis = 0) > 50] = np.array([[10], [20], [30], [40]])
        var y = x.clone()
        y[_, y.sum(axis = 0) >. 50] = [[10], [20], [30], [40]].toTensor()

        let exp = [[  4, 10, 10],
                  [  3, 20, 20],
                  [  1, 30, 30],
                  [  8, 40, 40]].toTensor()

        check: y == exp

      block: # y[y.sum(axis = 1) > 50, :] = np.array([-10, -20, -30])
        var y = x.clone()
        y[y.sum(axis = 1) >. 50, _] = [[-10, -20, -30]].toTensor()

        let exp = [[-10, -20, -30],
                  [-10, -20, -30],
                  [  1,   8,   7],
                  [  8,   6,   8]].toTensor()

        check: y == exp

    # TODO - only broadcastable tensor assign are supported at the moment
    # test "Masked axis assign multidimensional tensor via fancy indexing":
    #   block: # y[:, y.sum(axis = 0) > 50] = np.array([[10, 50], [20, 60], [30, 70], [40, 80]])
    #     var y = x.clone()
    #     y[_, y.sum(axis = 0) >. 50] = [[10, 50],
    #                                    [20, 60],
    #                                    [30, 70],
    #                                    [40, 80]].toTensor()
    #
    #     let exp = [[  4, 10, 50],
    #                [  3, 20, 60],
    #                [  1, 30, 70],
    #                [  8, 40, 80]].toTensor()
    #
    #     check: y == exp
    #
    #   block: # y[y.sum(axis = 1) > 50, :] = np.array([-10, -20, -30], [-40, -50, -60])
    #     var y = x.clone()
    #     y[y.sum(axis = 1) >. 50, _] = [[-10, -20, -30],
    #                                    [-40, -50, -60]].toTensor()
    #
    #     let exp = [[-10, -20, -30],
    #                [-40, -50, -60],
    #                [  1,   8,   7],
    #                [  8,   6,   8]].toTensor()
    #
    #     check: y == exp

main()
GC_fullCollect()
