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
import unittest, math

suite "[ML] Metrics":
  test "Accuracy score":
    let
      y_pred = [0, 2, 1, 3].toTensor
      y_true = [0, 1, 2, 3].toTensor

    check: accuracy_score(y_true, y_pred) == 0.5

  test "Mean absolute error":
    var y_true = [0.9, 0.2, 0.1, 0.4, 0.9].toTensor()
    var y =      [1.0, 0.0, 0.0, 1.0, 1.0].toTensor()
    check: y.absolute_error(y_true).sum() == 1.1
    check: y.mean_absolute_error(y_true) == (1.1 / 5.0)

  test "Relative error":
    var y_true = [0.0,  0.0, -1.0, 1e-8, 1e-8].toTensor()
    var y =      [0.0, -1.0,  0.0, 0.0,  1e-7].toTensor()
    check: y.relative_error(y_true) == [0.0, 1.0, 1.0, 1.0, 0.9].toTensor()
    check: y.mean_relative_error(y_true) == 0.78

  test "Mean squared error":
    var y_true = [0.9, 0.2, 0.1, 0.4, 0.9].toTensor()
    var y =      [1.0, 0.0, 0.0, 1.0, 1.0].toTensor()
    check: y.squared_error(y_true).sum() == 0.43
    check: y.mean_squared_error(y_true) == (0.43 / 5.0)
