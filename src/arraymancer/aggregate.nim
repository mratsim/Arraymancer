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

proc sum*[B; T: SomeNumber](t: Tensor[B,T]): T =
  # Compute the sum of all elements of T
  result = 0.T
  for val in t:
    result += val

proc sum*[B; T: SomeNumber](t: Tensor[B,T], axis: int): Tensor[B, T] =
  # Compute the sum of all elements of T along an axis
  var agg_shape = t.shape
  agg_shape[axis] = 1

  result = zeros(agg_shape, T, B)
  for t_slice in t.axis(axis):
    result += t_slice