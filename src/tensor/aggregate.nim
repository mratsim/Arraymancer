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

import  ./data_structure,
        ./operators_blas_l1

# ### Standard aggregate functions
# TODO consider using stats from Nim standard lib: https://nim-lang.org/docs/stats.html#standardDeviation,RunningStat

proc sum*[T: SomeNumber](t: Tensor[T]): T {.noSideEffect.}=
  ## Compute the sum of all elements of T

  result = 0.T
  for val in t:
    result += val

proc sum*[T: SomeNumber](t: Tensor[T], axis: int): Tensor[T] {.inline.}=
  ## Compute the sum of all elements of T along an axis
  proc sum_closure(r: var Tensor[T], x: Tensor[T]) =
    r += x
  t.reduce(sum_closure, axis)

proc mean*[T: SomeReal](t: Tensor[T]): T {.inline.}=
  ## Compute the mean of all elements of T
  t.sum / t.size.T

proc mean*[T: SomeReal](t: Tensor[T], axis: int): Tensor[T] {.inline.}=
  ## Compute the mean of T along an axis
  t.sum(axis) / t.shape[axis].T
