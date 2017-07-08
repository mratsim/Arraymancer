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

# # Compute aggregate/reduction/folds over tensors

# ### Elementwise generic aggregate functions
# Note: You can't pass builtins like `+` or `+=` due to Nim limitations
# https://github.com/nim-lang/Nim/issues/2172

proc agg*[B; T: SomeNumber](t: Tensor[B,T],
                            f:(T, T)-> T,
                            start_val: T
                            ): T {.noSideEffect.}=
  ## Compute the aggregate
  ## Input:
  ##     - A tensor to aggregate on
  ##     - The aggregation function. It is applied this way: new_aggregate = f(old_aggregate, current_value)
  ##     - The starting value
  ##     - The axis
  result = start_val
  for val in t:
    result = f(result, val)

proc agg_inplace*[B; T: SomeNumber](
                            accum_val: var T,
                            f: proc(x:var T, y:T), # We can't use the nice future syntax here for unknown reason
                            t: Tensor[B,T],
                            ) {.noSideEffect.}=
  ## Compute the aggregate
  ## Input:
  ##     - The accumulating value which will be modified in-place
  ##     - The aggregation in-place function. It is applied this way: f(var old_aggregate, current_value)
  ##     - A tensor to aggregate from
  ##     - The axis
  for val in t:
    f(accum_val, val)


# ### Axis generic functions
# `+`, `+=` for tensors are not "built-ins"

proc agg*[B; T: SomeNumber](t: Tensor[B,T],
                            f:(Tensor[B,T], Tensor[B,T])-> Tensor[B,T],
                            start_val: Tensor[B,T],
                            axis: int
                            ): Tensor[B,T] {.noSideEffect.}=
  ## Compute the aggregate along an axis
  ## Input:
  ##     - A tensor to aggregate on
  ##     - The aggregation function. It is applied this way: new_aggregate = f(old_aggregate, current_value)
  ##     - The starting value
  ##     - The axis

  result = start_val
  for val in t.axis(axis):
    result = f(result, val)

proc agg_inplace*[B; T: SomeNumber](
                            accum_val: var Tensor[B,T],
                            f: proc(x:var Tensor[B,T], y:Tensor[B,T]), # We can't use the nice future syntax here for unknown reason
                            t: Tensor[B,T],
                            axis: int
                            ) {.noSideEffect.}=
  ## Compute the aggregate along an axis
  ## Input:
  ##     - The accumulating value which will be modified in-place
  ##     - A tensor to aggregate from
  ##     - The aggregation in-place function. It is applied this way: f(var old_aggregate, current_value)
  ##     - The axis

  for val in t.axis(axis):
    f(accum_val, val)


# ### Standard aggregate functions

proc sum*[B; T: SomeNumber](t: Tensor[B,T]): T {.noSideEffect.}=
  ## Compute the sum of all elements of T
  # TODO tests
  result = 0.T
  for val in t:
    result += val

proc sum*[B; T: SomeNumber](t: Tensor[B,T], axis: int): Tensor[B, T] {.noSideEffect.}=
  ## Compute the sum of all elements of T along an axis
  # TODO tests
  var agg_shape = t.shape
  agg_shape[axis] = 1

  result = zeros(agg_shape, T, B)
  result.agg_inplace(`+=`, t, axis)

proc mean*[B; T: SomeReal](t: Tensor[B,T]): T {.noSideEffect.}=
  ## Compute the mean of all elements of T
  # TODO tests
  return t.sum / t.shape.product.T

proc mean*[B; T: SomeReal](t: Tensor[B,T], axis: int): Tensor[B, T] {.noSideEffect.}=
  ## Compute the mean of T along an axis
  # TODO tests
  let n = t.shape[axis]
  return t.sum(axis) / n.T