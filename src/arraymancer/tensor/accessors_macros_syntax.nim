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

import ../laser/dynamic_stack_arrays

# ## This file adds slicing syntactic sugar.
# ## Foo being:
# Tensor of shape 5x5 of type "int" on backend "Cpu"
# |1      1       1       1       1|
# |2      4       8       16      32|
# |3      9       27      81      243|
# |4      16      64      256     1024|
# |5      25      125     625     3125|
#
#
# ## Slicing
# Basic indexing - foo[2, 3]
# Basic indexing - foo[1+1, 2*2*1]
# Basic slicing - foo[1..2, 3]
# Basic slicing - foo[1+1..4, 3-2..2]
# Span slices - foo[_, 3]
# Span slices - foo[1.._, 3]
# Span slices - foo[_..3, 3]
# Span slices - foo[_.._, 3]
# Stepping - foo[1..3|2, 3]
# Span stepping - foo[_.._|2, 3]
# Span stepping - foo[_.._|+2, 3]
# Span stepping - foo[1.._|1, 2..3]
# Span stepping - foo[_..<4|2, 3]
# Slicing until at n from the end - foo[0..^4, 3]
# Span Slicing until at n from the end - foo[_..^2, 3]
# Stepped Slicing until at n from the end - foo[1..^1|2, 3]
# Slice from the end - foo[^1..0|-1, 3]
# Slice from the end - expect non-negative step error - foo[^1..0, 3]
# Slice from the end - foo[^(2*2)..2*2, 3]
# Slice from the end - foo[^3..^2, 3]

# ## Assignement
# Slice to a single value - foo[1..2, 3..4] = 999
# Slice to an array/seq of values - foo[0..1,0..1] = [[111, 222], [333, 444]]
# Slice to values from a view/Tensor - foo[^2..^1,2..4] = bar
# Slice to values from view of same Tensor - foo[^2..^1,2..4] = foo[^1..^2|-1, 4..2|-1]


type SteppedSlice* = object
  ## Internal: A slice object related to a tensor single dimension:
  ##   - a, b: Respectively the beginning and the end of the range of the dimension
  ##   - step: The stepping of the slice (can be negative)
  ##   - a/b_from_end: Indicates if a/b should be counted from 0 or from the end of the tensor relevant dimension.
  ## Slicing syntax like a[2, 1..<5, _] will be converted at compile-time to SteppedSlices
  a*, b*: int
  step*: int
  a_from_end*: bool
  b_from_end*: bool

type Step* = object
  ## Internal: Workaround to build ``SteppedSlice`` without using parenthesis.
  ##
  ## Expected syntax is ``tensor[0..10|1]``.
  ##
  ## Due to operator precedence of ``|`` over ``..`` [0..10|1] is interpreted as [0..(10|1)]
  b: int
  step: int

# Following https://github.com/mratsim/Arraymancer/issues/61 and
# https://github.com/mratsim/Arraymancer/issues/43 we export _ directly
const _* = SteppedSlice(b: 1, step: 1, b_from_end: true)

type Ellipsis* = object ##Dummy type for ellipsis i.e. "Don't slice the rest of dimensions"

const `...`* = Ellipsis()

# #########################################################################
# Collection of slices

type ArrayOfSlices* = DynamicStackArray[SteppedSlice]

proc toArrayOfSlices*(s: varargs[SteppedSlice]): ArrayOfSlices {.inline.} =
  # boundsChecks automatically done for array indexing
  # when compileOption("boundChecks"):
  #   assert s.len <= MAXRANK
  result.len = s.len
  for i in 0..<s.len:
    result.data[i] = s[i]

proc initSpanSlices*(len: int): ArrayOfSlices {.inline.} =
  result.len = len
  for i in 0..<len:
    result.data[i] = SteppedSlice(b: 1, step: 1, b_from_end: true)



# #########################################################################
# Slicing notation

# Procs to manage all integer, slice, SteppedSlice
proc `|`*(s: Slice[int], step: int): SteppedSlice {.noSideEffect, inline.}=
  ## Internal: A ``SteppedSlice`` constructor
  ## Input:
  ##     - a slice
  ##     - a step
  ## Returns:
  ##     - a ``SteppedSlice``
  return SteppedSlice(a: s.a, b: s.b, step: step)

proc `|`*(b, step: int): Step {.noSideEffect, inline.}=
  ## Internal: A ``Step`` constructor
  ##
  ## ``Step`` is a workaround due to operator precedence.
  ##
  ## [0..10|1] is interpreted as [0..(10|1)]
  ## Input:
  ##     - the end of a slice range
  ##     - a step
  ## Returns:
  ##     - a ``Step``
  return Step(b: b, step: step)

proc `|`*(ss: SteppedSlice, step: int): SteppedSlice {.noSideEffect, inline.}=
  ## Internal: Modifies the step of a ``SteppedSlice``
  ## Input:
  ##     - a ``SteppedSLice``
  ##     - the new stepping
  ## Returns:
  ##     - a ``SteppedSLice``
  result = ss
  result.step = step

proc `|+`*(s: Slice[int], step: int): SteppedSlice {.noSideEffect, inline.}=
  ## Internal: Alias for ``|``
  return `|`(s, step)

proc `|+`*(b, step: int): Step {.noSideEffect, inline.}=
  ## Internal: Alias for ``|``
  return `|`(b, step)

proc `|+`*(ss: SteppedSlice, step: int): SteppedSlice {.noSideEffect, inline.}=
  ## Internal: Alias for ``|``
  return `|`(ss, step)

proc `|-`*(s: Slice[int], step: int): SteppedSlice {.noSideEffect, inline.}=
  ## Internal: A ``SteppedSlice`` constructor
  ##
  ## Workaround to tensor[slice|-1] being interpreted as [slice `|-` 1]
  ##
  ## Properly create ``SteppedSLice`` with negative stepping
  return SteppedSlice(a: s.a, b: s.b, step: -step)

proc `|-`*(b, step: int): Step {.noSideEffect, inline.}=
  ## Internal: A ``SteppedSlice`` constructor
  ##
  ## Workaround to tensor[0..10|-1] being intepreted as [0 .. (10 `|-` 1)]
  ##
  ## Properly create ``SteppedSLice`` with negative stepping
  return Step(b: b, step: -step)

proc `|-`*(ss: SteppedSlice, step: int): SteppedSlice {.noSideEffect, inline.}=
  ## Internal: Modifies the step of a ``SteppedSlice``
  ##
  ## Workaround to tensor[slice|-1] being interpreted as [slice `|-` 1]
  ##
  ## Properly create ``SteppedSLice`` with negative stepping
  result = ss
  result.step = -step

proc `..`*(a: int, s: Step): SteppedSlice {.noSideEffect, inline.} =
  ## Internal: Build a SteppedSlice from [a .. (b|step)] (workaround to operator precedence)
  ## Input:
  ##     - the beginning of the slice range
  ##     - a ``Step`` workaround object
  ## Returns:
  ##     - a ``SteppedSlice``, end of range will be inclusive
  return SteppedSlice(a: a, b: s.b, step: s.step)

proc `..<`*(a: int, s: Step): SteppedSlice {.noSideEffect, inline.} =
  ## Internal: Build a SteppedSlice from [a ..< (b|step)] (workaround to operator precedence)
  ## Input:
  ##     - the beginning of the slice range
  ##     - a ``Step`` workaround object
  ## Returns:
  ##     - a ``SteppedSlice``, end of range will be exclusive.
  return SteppedSlice(a: a, b: s.b - 1, step: s.step)

proc `..^`*(a: int, s: Step): SteppedSlice {.noSideEffect, inline.} =
  ## Internal: Build a SteppedSlice from [a ..^ (b|step)] (workaround to operator precedence and ..^b not being interpreted as .. ^b)
  ## Input:
  ##     - the beginning of the slice range
  ##     - a ``Step`` workaround object
  ## Returns:
  ##     - a ``SteppedSlice``, end of range will start at "b" away from the end
  return SteppedSlice(a: a, b: s.b, step: s.step, b_from_end: true)

proc `^`*(s: SteppedSlice): SteppedSlice {.noSideEffect, inline.} =
  ## Internal: Prefix to a to indicate starting the slice at "a" away from the end
  ## Note: This does not automatically inverse stepping, what if we want ^5..^1
  result = s
  result.a_from_end = not result.a_from_end

proc `^`*(s: Slice): SteppedSlice {.noSideEffect, inline.} =
  ## Internal: Prefix to a to indicate starting the slice at "a" away from the end
  ## Note: This does not automatically inverse stepping, what if we want ^5..^1
  return SteppedSlice(a: s.a, b: s.b, step: 1, a_from_end: true)
