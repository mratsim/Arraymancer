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


import  ../data_structure,
        future

proc fmap*[T, U](t: Tensor[T], f: T -> U): Tensor[U]
  {.deprecated, inline.}=
  ## DEPRECATED
  ##
  ## Replace by map2
  t.map(f)

proc fmap2*[T, U, V](t1: Tensor[T], t2: Tensor[U], f: (T,U) -> V): Tensor[V]
  {.deprecated, inline.}=
  ## DEPRECATED
  ##
  ## Replaced by map2
  ##
  ## Note the new argument order of map2 to accomodate for
  ## t1.map2(`op`, t2) where op is an infix operator.
  t1.map2(f, t2)


# # Compute aggregate/reduction/folds over tensors

# ### Elementwise generic aggregate functions
# Note: You can't pass builtins like `+` or `+=` due to Nim limitations
# https://github.com/nim-lang/Nim/issues/2172

proc agg*[T: SomeNumber](t: Tensor[T],
                            f:(T, T)-> T,
                            start_val: T
                            ): T
  {.noSideEffect, inline, deprecated.}=
  ## DEPRECATED, use fold instead.
  ##
  ## Note: order between function f and start_val has changed
  ##
  ## Compute the aggregate
  ## Input:
  ##     - A tensor to aggregate on
  ##     - The aggregation function. It is applied this way: new_aggregate = f(old_aggregate, current_value)
  ##     - The starting value
  ##     - The axis
  t.fold(start_val, f)

proc agg_inplace*[T: SomeNumber](
                            accum_val: var T,
                            f: proc(x:var T, y:T), # We can't use the nice future syntax here for unknown reason
                            t: Tensor[T],
                            )
  {.noSideEffect, inline, deprecated.}=
  ## DEPRECATED, use fold instead.
  ##
  ## You will have to switch to a non-inplace function.
  ##
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

proc agg*[T: SomeNumber](t: Tensor[T],
                            f:(Tensor[T], Tensor[T])-> Tensor[T],
                            start_val: Tensor[T],
                            axis: int
                            ): Tensor[T]
  {.noSideEffect, inline, deprecated.}=
  ## DEPRECATED, use fold instead.
  ##
  ## Note: order between function f and start_val has changed
  ##
  ## Input:
  ##     - A tensor to aggregate on
  ##     - The aggregation function. It is applied this way: new_aggregate = f(old_aggregate, current_value)
  ##     - The starting value
  ##     - The axis

  t.fold(start_val, f, axis)

proc agg_inplace*[T: SomeNumber](
                            accum_val: var Tensor[T],
                            f: proc(x:var Tensor[T], y:Tensor[T]), # We can't use the nice future syntax here for unknown reason
                            t: Tensor[T],
                            axis: int
                            )
  {.noSideEffect, inline, deprecated.}=
  ## DEPRECATED, use fold instead.
  ##
  ## You will have to switch to a non-inplace function.
  ##
  ## Input:
  ##     - The accumulating value which will be modified in-place
  ##     - A tensor to aggregate from
  ##     - The aggregation in-place function. It is applied this way: f(var old_aggregate, current_value)
  ##     - The axis

  for val in t.axis(axis):
    f(accum_val, val)
