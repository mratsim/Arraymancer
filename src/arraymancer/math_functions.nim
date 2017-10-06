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


# Non-operator math functions

proc reciprocal*[T: SomeReal](t: Tensor[T]): Tensor[T] =
  # Return a tensor with the reciprocal 1/x of all elements
  t.mapT(1.T/x)

proc mreciprocal*[T: SomeReal](t: var Tensor[T]) =
  # Apply the reciprocal 1/x in-place to all elements of the Tensor
  t.applyT(1.T/x)

proc negate*[T: SomeSignedInt|SomeReal](t: Tensor[T]): Tensor[T] =
  # Return a tensor with all elements negated (10 -> -10)
  t.mapT(-x)

proc mnegate*[T: SomeSignedInt|SomeReal](t: var Tensor[T]) =
  # Negate in-place all elements of the tensor (10 -> -10)
  t.applyT(-x)

proc `-`*[T: SomeNumber](t: Tensor[T]): Tensor[T] =
  ## Negate all values of a Tensor
  t.mapT(-x)