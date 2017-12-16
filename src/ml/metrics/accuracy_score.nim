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

import ../../tensor/tensor


proc accuracy_score*[T](y_true, y_pred: Tensor[T]): float =
  ## Input:
  ##   - y_true: Tensor[T] containing the ground truth (correct) labels
  ##   - y_pred: Tensor[T] containing the predicted labels
  ##
  ## Returns:
  ##   - The proportion of correctly classified samples (as float).

  result = (y_true .== y_pred).astype(float).mean